#!/usr/bin/env python3
"""Phase 5 held-out test (PROTOCOL.md Section 5) and estimator audit (Section 6).

Every selected and final checkpoint (``checkpoints/MANIFEST.txt``) x 6 presets x test seeds
5001, 5002 x 1024 environments x 30 s, deterministic, Mjlab-Leap-HwLim-<cond>.  Estimator B
is scored closed-loop on every nominal rollout (audit only).  Several workers may run at
once (per-item claim files); finished items are skipped, so the script is resumable.

    python experiments/hwlimit_ac/scripts/test_eval.py            # one worker
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

E = Path("experiments/hwlimit_ac")
S = Path("experiments/scratch_state_estimation")
TEST_SEEDS = (5001, 5002)
PRESETS = ("nominal", "heavy_slippery", "light_grippy", "high_delay", "weak_motors", "obs_noise_x3")


def items():
  ck = [l.split(" | ")[0] for l in (E / "checkpoints" / "MANIFEST.txt").read_text().splitlines() if not l.startswith("#")]
  # selected checkpoints and nominal first, so the primary result finishes first
  ck.sort(key=lambda n: (0 if "_selected_" in n else 1, n))
  out = []
  for preset in PRESETS:
    for seed in TEST_SEEDS:
      for c in ck:
        out.append((c, preset, seed))
  out.sort(key=lambda t: (0 if "_selected_" in t[0] else 1, PRESETS.index(t[1]), t[2]))
  return out


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--num-envs", type=int, default=1024)
  p.add_argument("--device", default="cuda:0")
  a = p.parse_args()

  from in_hand_rotation_mjlab.estimation.model import load_estimator
  from in_hand_rotation_mjlab.hwlimit_study.evaluation import EvalConfig, evaluate, summarize

  est = load_estimator(str(S / "models" / "estimator_B.pt"), map_location=a.device).to(a.device).eval()
  out = E / "results" / "test"
  (out / ".claims").mkdir(parents=True, exist_ok=True)
  for ck, preset, seed in items():
    key = f"{ck.removesuffix('.pt')}__{preset}__seed{seed}"
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
    task = f"Mjlab-Leap-HwLim-{ck[0]}"
    pe, extra = evaluate(task, str(E / "checkpoints" / ck), seed, preset,
                         EvalConfig(num_envs=a.num_envs, num_steps=600, device=a.device),
                         estimator=est if preset == "nominal" else None)
    np.savez_compressed(out / f"{key}.npz", **pe)
    (out / f"{key}.json").write_text(json.dumps({"checkpoint": ck, "preset": preset, "seed": seed, "task": task,
                                                  "summary": summarize(pe), "extra": extra,
                                                  "host": os.uname().nodename, "seconds": round(time.time() - t0, 1)}, indent=1))
    s = summarize(pe)
    print(f"[{key}] turns {s['turns']['mean']:.3f} fail {s['failure']['mean']:.3f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
  main()
