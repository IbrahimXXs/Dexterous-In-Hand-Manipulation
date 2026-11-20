#!/usr/bin/env python3
"""Worker for Grasp-Rotation-V2 rollouts (simulation only).  Items are JSON objects in a list
file; several workers share a list through claim files.  Each item is saved as
data/<key>.npz (rollout.rollout output) plus results/items/<key>.json (summary).

Item fields: key, policy (A1|A2|A3|R1|hold|<path>), seed, num_envs, steps, preset,
assign (null = uniform cache draw | "all" = env i -> grasp i mod 239 |
"grasps:<g1,g2,...>" = env i -> g[i mod len]), pose_range (null or dict), record_actions.

    python experiments/grasp_rotation_v2/scripts/run_items.py --items experiments/grasp_rotation_v2/items/p1_dev.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

E = Path("experiments/grasp_rotation_v2")
POLICIES = {"A1": "experiments/hwlimit_ac/checkpoints/A_seed1_selected_model_2600.pt",
            "A2": "experiments/hwlimit_ac/checkpoints/A_seed2_selected_model_2900.pt",
            "A3": "experiments/hwlimit_ac/checkpoints/A_seed3_selected_model_2600.pt",
            "R1": "experiments/retention_v1/checkpoints/R1_selected_model_2600.pt",
            "R1final": "experiments/retention_v1/checkpoints/R1_final_model_3000.pt", "hold": None}
N_GRASPS = 239


def assignment(spec, n):
  if spec is None:
    return None
  if spec == "all":
    return np.arange(n) % N_GRASPS
  if spec.startswith("grasps:"):
    g = np.array([int(x) for x in spec.split(":", 1)[1].split(",")])
    return g[np.arange(n) % len(g)]
  raise ValueError(spec)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--items", required=True)
  p.add_argument("--device", default="cuda:0")
  a = p.parse_args()
  from in_hand_rotation_mjlab.grasp_rotation_v2.rollout import rollout, summary_per_env

  items = json.loads(Path(a.items).read_text())
  (E / "results" / "items" / ".claims").mkdir(parents=True, exist_ok=True)
  for it in items:
    key = it["key"]
    done = E / "results" / "items" / f"{key}.json"
    if done.exists():
      continue
    claim = E / "results" / "items" / ".claims" / key
    if claim.exists() and time.time() - claim.stat().st_mtime > 3 * 3600:
      claim.unlink(missing_ok=True)
    try:
      os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
      continue
    t0 = time.time()
    pol = it["policy"]
    ck = POLICIES[pol] if pol in POLICIES else pol
    task = it.get("task", "Mjlab-Leap-RetV1-A")
    tr = rollout(task, ck, it["seed"], it["num_envs"], it["steps"], a.device, preset=it.get("preset", "nominal"),
                 assignment=assignment(it.get("assign"), it["num_envs"]), pose_range=it.get("pose_range"),
                 record_actions=it.get("record_actions", False), terminations_off=tuple(it.get("terminations_off", ())),
                 record_quat=it.get("record_quat", False))
    np.savez_compressed(E / "data" / f"{key}.npz", **tr)
    summ = {}
    for steps, tag in ((600, "30s"), (it["steps"], f"{it['steps'] * 0.05:.0f}s")):
      s = summary_per_env(tr, min(steps, it["steps"]))
      summ[tag] = {k: float(v.mean()) for k, v in s.items()}
    done.write_text(json.dumps({"item": it, "summary": summ, "host": os.uname().nodename,
                                "seconds": round(time.time() - t0, 1)}, indent=1))
    s30 = summ["30s"]
    print(f"[{key}] 30s turns {s30['turns']:.3f} fail {s30['failure']:.3f} surv {s30['survival_s']:.1f} "
          f"yaw {s30['yaw_rate']:.3f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
  main()
