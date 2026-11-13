#!/usr/bin/env python3
"""Standalone inference example: 320-D proprioceptive history -> [thumb, index, middle, ring] force (N).

No simulator, no policy, no hardware access.  Uses held-out test observations
from the force dataset if present, otherwise a zero observation.

    python experiments/force_estimation/scripts/inference_example.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from in_hand_rotation_mjlab.force_estimation.inference import ForceEstimatorRuntime

CKPT = "experiments/force_estimation/models/force_estimator.pt"


def main():
  est = ForceEstimatorRuntime(CKPT, device="cpu")
  shards = sorted(Path("experiments/force_estimation/data/test").glob("*/obs.npy"))
  if shards:
    obs = np.load(shards[0], mmap_mode="r")[100, :8]  # 8 envs at control step 100
    true = np.load(shards[0].parent / "force_mean.npy")[100, :8]
  else:
    obs, true = np.zeros((8, 320), np.float32), None
  forces = est.predict(obs)  # (8, 4)
  print("output order:", est.output_names)
  for i in range(len(forces)):
    line = f"env {i}: estimate " + " ".join(f"{f:6.3f}" for f in forces[i])
    if true is not None:
      line += "   true " + " ".join(f"{f:6.3f}" for f in true[i])
    print(line)
  single = est.predict(obs[0])  # (4,)
  assert single.shape == (4,) and np.all(np.isfinite(single)) and np.all(single >= 0)
  t0 = time.perf_counter()
  for _ in range(1000):
    est.predict(obs[0])
  print(f"CPU latency per single call (incl. numpy<->torch): {(time.perf_counter() - t0):.3f} ms")


if __name__ == "__main__":
  main()
