#!/usr/bin/env python3
"""60-s retention test (PROTOCOL.md Section 3): full first-episode traces, seed 5001,
presets nominal / high_delay / heavy_slippery, 1024 environments x 1200 steps.

    python experiments/retention_v1/scripts/test60.py --policies A1 A2 A3      # baseline
    python experiments/retention_v1/scripts/test60.py --policies R1            # after selection
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

E = Path("experiments/retention_v1")
A_CK = {"A1": "experiments/hwlimit_ac/checkpoints/A_seed1_selected_model_2600.pt",
        "A2": "experiments/hwlimit_ac/checkpoints/A_seed2_selected_model_2900.pt",
        "A3": "experiments/hwlimit_ac/checkpoints/A_seed3_selected_model_2600.pt"}
PRESETS = ("nominal", "high_delay", "heavy_slippery")


def checkpoint(name: str, kind: str) -> tuple[str, str]:
  if name.startswith("A"):
    return "Mjlab-Leap-HwLim-A", A_CK[name]
  sel = json.loads((E / "results" / "convergence.json").read_text())["runs"][name]
  it = sel["selected_iteration"] if kind == "selected" else 3000
  return "Mjlab-Leap-RetV1-A", str(E / "checkpoints" / f"{name}_{kind}_model_{it}.pt")


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--policies", nargs="+", required=True)
  p.add_argument("--kind", default="selected", choices=["selected", "final"])
  p.add_argument("--seed", type=int, default=5001)
  p.add_argument("--num-envs", type=int, default=1024)
  p.add_argument("--device", default="cuda:0")
  a = p.parse_args()
  from in_hand_rotation_mjlab.retention_v1.traces import record_traces

  for name in a.policies:
    task, ck = checkpoint(name, a.kind)
    for preset in PRESETS:
      tag = name if name.startswith("A") else f"{name}_{a.kind}"
      out = E / "data" / f"test60_{tag}_{preset}_seed{a.seed}.npz"
      if out.exists():
        continue
      t0 = time.time()
      tr = record_traces(task, ck, a.seed, a.num_envs, 1200, a.device, preset)
      np.savez_compressed(out, **{k: (v.astype(np.float16) if isinstance(v, np.ndarray) and v.dtype == np.float32 and v.ndim >= 3 else v)
                                  for k, v in tr.items()})
      t30 = tr["yaw_inc"][:600].sum(0).mean() / (2 * np.pi)
      t60 = tr["yaw_inc"].sum(0).mean() / (2 * np.pi)
      print(f"[{tag} {preset}] turns30 {t30:.3f} turns60 {t60:.3f} retain60 {(tr['end_step'] < 0).mean():.3f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
  main()
