#!/usr/bin/env python3
"""Phase 1: record 60-s diagnostic traces of the hwlimit_ac A policies (simulation only).

Selected checkpoints of hwlimit A seeds 1-3 (read-only copies in
experiments/hwlimit_ac/checkpoints/), Mjlab-Leap-HwLim-A, nominal preset, diagnostic seed
9201 (used nowhere else), 1024 environments, 1200 control steps (60 s).  The 30-s view is
the first 600 steps of the same rollouts.  Output: data/diag_<run>.npz (git-ignored).

    python experiments/retention_v1/scripts/record_diagnosis.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

E = Path("experiments/retention_v1")
H = Path("experiments/hwlimit_ac/checkpoints")
POLICIES = {"A1": "A_seed1_selected_model_2600.pt", "A2": "A_seed2_selected_model_2900.pt", "A3": "A_seed3_selected_model_2600.pt"}


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--seed", type=int, default=9201)
  p.add_argument("--num-envs", type=int, default=1024)
  p.add_argument("--steps", type=int, default=1200)
  p.add_argument("--preset", default="nominal")
  p.add_argument("--device", default="cuda:0")
  a = p.parse_args()
  from in_hand_rotation_mjlab.estimation.model import load_estimator
  from in_hand_rotation_mjlab.retention_v1.traces import record_traces, turns_until

  est = load_estimator("experiments/scratch_state_estimation/models/estimator_B.pt", map_location=a.device).to(a.device).eval()

  (E / "data").mkdir(parents=True, exist_ok=True)
  for run, ck in POLICIES.items():
    out = E / "data" / f"diag_{run}_{a.preset}_seed{a.seed}.npz"
    if out.exists():
      continue
    t0 = time.time()
    tr = record_traces("Mjlab-Leap-HwLim-A", str(H / ck), a.seed, a.num_envs, a.steps, a.device, a.preset, estimator=est)
    np.savez_compressed(out, **{k: (v.astype(np.float16) if isinstance(v, np.ndarray) and v.dtype == np.float32 and v.ndim >= 3 else v)
                                for k, v in tr.items()})
    alive_end = tr["end_step"] < 0
    print(f"[{run}] 30 s turns {turns_until(tr, 600).mean():.3f}  60 s turns {turns_until(tr, a.steps).mean():.3f}  "
          f"survive 30 s {(tr['end_step'] < 0).mean() if a.steps <= 600 else ((tr['end_step'] < 0) | (tr['end_step'] >= 600)).mean():.3f}  "
          f"survive 60 s {alive_end.mean():.3f}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
  main()
