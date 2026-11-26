#!/usr/bin/env python3
"""C5-HW-V2 evaluation worker (simulation only).

The rollout and the metrics are C5-HW-EFFORT-V1's (``c5_hw_v1/scripts/eval_items.py``: ``rollout``,
``summarise``, jam proxy), imported unchanged.  Added here:

* setting **K** ("kit start"): episodes start from ``cache/kit_start_cache.npz`` (FIT_PLAN.md B; environment i
  gets entry i mod N; the stored joint target is the measured post-ramp pose, as the kit resets it);
* actuator **hwmatched**: the 350 mA-equivalent effort range plus the fitted response randomisation of
  ``results/response_fit.json`` (only if the fit passed; otherwise this actuator is not defined).

Items (setting, policy, actuator, preset, seed) are claimed by several workers through claim files.

    python experiments/c5_hw_v2/scripts/eval_items.py --make-items step_b
    python experiments/c5_hw_v2/scripts/eval_items.py --items experiments/c5_hw_v2/items/step_b.json
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
E = PROJECT / "experiments" / "c5_hw_v2"
KIT_CACHE = E / "cache" / "kit_start_cache.npz"


def _load(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[name] = mod
  spec.loader.exec_module(mod)
  return mod


V1 = _load("c5_hw_v1_eval_items", PROJECT / "experiments" / "c5_hw_v1" / "scripts" / "eval_items.py")
RF = _load("c5_hw_v2_response_fit", E / "scripts" / "response_fit.py")

ACTUATOR_TASK = {"default": "Mjlab-Leap-RetV1-A", "effort": "Mjlab-Leap-C5HW-EFFORT", "hwmatched": "Mjlab-Leap-C5HW-EFFORT"}
SETTINGS = {**V1.SETTINGS, "K": "kit"}


def policy_path(p: str) -> str:
  if p.startswith("C5HW2_"):
    c = sorted(glob.glob(str(E / "checkpoints" / f"{p}_selected_model_*.pt")))
    assert len(c) == 1, c
    return c[0]
  return V1.policy_path(p)


def hwmatched_ranges() -> dict:
  r = json.loads((E / "results" / "response_fit.json").read_text())
  if r["primary"]["status"] != "ok":
    raise RuntimeError(f"response fit status {r['primary']['status']}: no hardware-matched actuator")
  f = r["fitted_randomisation"]
  return {"delay": tuple(f["lag_range_steps"]), "damping": tuple(f["damping_scale_range"]),
          "friction": tuple(f["friction_scale_range"])}


@contextlib.contextmanager
def setup(actuator: str, setting: str):
  """Patch the task config (hwmatched) and the reset cache (K) for one rollout; nothing on disk changes."""
  from in_hand_rotation_mjlab.grasp_rotation_v2 import rollout as R

  base_make = R.make_env

  def make_env(task, *a, **k):
    if setting == "K":
      k["grasp_cache"] = str(KIT_CACHE)
    return base_make(task, *a, **k)

  ranges = hwmatched_ranges() if actuator == "hwmatched" else {}
  R.make_env = make_env
  try:
    with RF.patched_events(**ranges):
      yield
  finally:
    R.make_env = base_make


def assignment(setting: str, spec, n: int):
  if setting == "K":
    m = len(np.load(KIT_CACHE)["cube_size"])
    return np.arange(n) % m
  return V1.assignment(spec, n)


def make_items(name: str) -> list[dict]:
  items = []
  if name == "step_b":  # FIT_PLAN.md B: survival from kit starts vs the standard grasp-49 start, validation seed
    for pol in ("C5_1", "C5_2", "C5_3", "EFFORT_1"):
      for act in ("default", "effort"):
        for st in ("K", "B"):
          items.append({"key": f"stepB_{st}_{pol}_{act}_nominal_s7001", "setting": st, "policy": pol, "actuator": act,
                        "seed": 7001, "preset": "nominal", "num_envs": 2048, "steps": 600})
    return items
  if name == "heldout_baselines_K":  # PROTOCOL.md: the kit-start setting for every frozen policy, held-out seeds
    for pol in ("A1", "A2", "A3", "R1", "C5_1", "C5_2", "C5_3", "EFFORT_1"):
      for act in ("default", "effort"):
        for seed in (6201, 6202):
          items.append({"key": f"K_{pol}_{act}_nominal_s{seed}", "setting": "K", "policy": pol, "actuator": act,
                        "seed": seed, "preset": "nominal", "num_envs": 2048, "steps": 1200})
    return items
  if name == "heldout_new":  # PROTOCOL.md: the new policy in every setting (A/B/C as C5-HW-EFFORT-V1, plus K)
    for act in ("default", "effort"):
      for st in ("A", "B", "C", "K"):
        runs = [(s, "nominal") for s in (6201, 6202)]
        if st == "A":
          runs += [(6201, p) for p in ("heavy_slippery", "high_delay", "obs_noise_x3")]
        for seed, preset in runs:
          items.append({"key": f"{st}_C5HW2_1_{act}_{preset}_s{seed}", "setting": st, "policy": "C5HW2_1", "actuator": act,
                        "seed": seed, "preset": preset, "num_envs": 2048, "steps": 1200})
    return items
  raise KeyError(name)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--items")
  ap.add_argument("--make-items")
  ap.add_argument("--device", default="cuda:0")
  a = ap.parse_args()
  if a.make_items:
    out = E / "items" / f"{a.make_items}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(make_items(a.make_items), indent=1))
    print("wrote", out)
    return
  import torch

  items = json.loads(Path(a.items).read_text())
  (E / "results" / "items" / ".claims").mkdir(parents=True, exist_ok=True)
  (E / "data" / "items").mkdir(parents=True, exist_ok=True)
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
    ck = policy_path(it["policy"])
    sha = hashlib.sha256(Path(ck).read_bytes()).hexdigest()
    st = it["setting"]
    with setup(it["actuator"], st):
      tr = V1.rollout(ACTUATOR_TASK[it["actuator"]], ck, it["seed"], it["num_envs"], it["steps"], a.device, it["preset"],
                      assignment(st, SETTINGS.get(st), it["num_envs"]))
    np.savez_compressed(E / "data" / "items" / f"{key}.npz", **tr)
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    summ = V1.summarise(tr)
    extra = {}
    if st == "K":
      extra["kit_cache_sha256"] = hashlib.sha256(KIT_CACHE.read_bytes()).hexdigest()
    if it["actuator"] == "hwmatched":
      extra["hwmatched_ranges"] = {k: list(v) for k, v in hwmatched_ranges().items()}
    done.write_text(json.dumps({"item": it, "task": ACTUATOR_TASK[it["actuator"]], "checkpoint": ck, "sha256": sha,
                                "summary": summ, **extra, "host": os.uname().nodename,
                                "seconds": round(time.time() - t0, 1)}, indent=1))
    s = summ["30s"]
    print(f"[{key}] 30s survival {s['completed']:.3f} turns {s['turns']:.3f} fail {s['failure']:.3f} "
          f"jam {s['jam_proxy']:.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
  main()
