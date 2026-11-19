#!/usr/bin/env python3
"""30-s held-out test of Retention-V1 (PROTOCOL.md Section 3), identical to the
hardware-limited A test (experiments/hwlimit_ac/scripts/test_eval.py): selected and final
checkpoints x 6 presets x seeds 5001, 5002 x 1024 environments x 30 s, deterministic,
``hwlimit_study.evaluation.evaluate``.  The A results are read from
experiments/hwlimit_ac/results/test/ (same code path, seeds and environments).

    python experiments/retention_v1/scripts/test30.py --runs R1
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

E = Path("experiments/retention_v1")
TEST_SEEDS = (5001, 5002)
PRESETS = ("nominal", "heavy_slippery", "light_grippy", "high_delay", "weak_motors", "obs_noise_x3")


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--runs", nargs="+", default=["R1"])
  p.add_argument("--num-envs", type=int, default=1024)
  p.add_argument("--device", default="cuda:0")
  a = p.parse_args()
  from in_hand_rotation_mjlab.hwlimit_study.evaluation import EvalConfig, evaluate, summarize

  conv = json.loads((E / "results" / "convergence.json").read_text())["runs"]
  out = E / "results" / "test"
  (out / ".claims").mkdir(parents=True, exist_ok=True)
  items = []
  for kind in ("selected", "final"):
    for preset in PRESETS:
      for seed in TEST_SEEDS:
        for r in a.runs:
          it = conv[r]["selected_iteration"] if kind == "selected" else 3000
          items.append((f"{r}_{kind}_model_{it}", preset, seed))
  for ck, preset, seed in items:
    key = f"{ck}__{preset}__seed{seed}"
    if (out / f"{key}.json").exists():
      continue
    claim = out / ".claims" / key
    if claim.exists() and time.time() - claim.stat().st_mtime > 1800:
      claim.unlink(missing_ok=True)
    try:
      os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
      continue
    t0 = time.time()
    pe, extra = evaluate("Mjlab-Leap-RetV1-A", str(E / "checkpoints" / f"{ck}.pt"), seed, preset,
                         EvalConfig(num_envs=a.num_envs, num_steps=600, device=a.device))
    np.savez_compressed(out / f"{key}.npz", **pe)
    s = summarize(pe)
    (out / f"{key}.json").write_text(json.dumps({"checkpoint": ck, "preset": preset, "seed": seed, "summary": s, "extra": extra,
                                                  "host": os.uname().nodename, "seconds": round(time.time() - t0, 1)}, indent=1))
    print(f"[{key}] turns {s['turns']['mean']:.3f} fail {s['failure']['mean']:.3f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
  main()
