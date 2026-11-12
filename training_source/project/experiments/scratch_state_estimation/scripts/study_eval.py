#!/usr/bin/env python3
"""Evaluation CLI for the from-scratch study.

  curve     validation evaluation of a run's checkpoints (learning curve in
            genuine turns + checkpoint selection).  Validation seed(s) are
            disjoint from the test seeds.
  final     held-out test evaluation of selected checkpoints on all presets.
  ablation  C / D feature ablation (normal / zero / mean / shuffle) with
            matched seeds, plus executed-action sensitivity on visited states.

Seeds: validation 7001; test 5001, 5002 (never used for selection).
Selection rule (fixed in advance): highest mean uninterrupted turns in 30 s on
the validation seed, nominal preset; ties broken by the later checkpoint.

    python study_eval.py curve --run logs/rsl_rl/scratch_A/<dir> --cond A --out results/curves
    python study_eval.py final --policy A_s1=A:<ckpt> ... --out results/final
    python study_eval.py ablation --policy C_s1=C:<ckpt> --out results/ablation
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from in_hand_rotation_mjlab.estimation.model import load_estimator
from in_hand_rotation_mjlab.scratch_study.evaluation import PRESETS, EvalConfig, evaluate, summarize

VAL_SEEDS = (7001,)
TEST_SEEDS = (5001, 5002)


def ckpts(run: str) -> dict[int, str]:
  out = {}
  for f in glob.glob(str(Path(run) / "model_*.pt")):
    out[int(re.findall(r"model_(\d+)\.pt", f)[0])] = f
  return dict(sorted(out.items()))


def cmd_curve(a):
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  name = a.name or Path(a.run).name
  dest = out / f"{name}.json"
  res = json.loads(dest.read_text()) if dest.exists() else {"run": a.run, "cond": a.cond, "points": {}}
  avail = ckpts(a.run)
  its = sorted(avail) if not a.iterations else [i for i in a.iterations if i in avail]
  if a.include_last and avail:
    its = sorted(set(its) | {max(avail)})
  for it in its:
    if str(it) in res["points"]:
      continue
    t0 = time.time()
    pe, _ = evaluate(a.cond, avail[it], VAL_SEEDS[0], "nominal", EvalConfig(num_envs=a.num_envs, num_steps=600))
    s = summarize(pe)
    res["points"][str(it)] = {k: s[k] for k in ("turns", "turns_with_resets", "positive_rotation_rad", "mean_yaw_rate", "survival_s",
                                                "failure", "drop", "pose_failure", "reward_sum", "drift_mean_mm", "tilt_mean_deg", "exec_saturation")}
    print(f"[curve {name}] it {it}: turns {s['turns']['mean']:.3f} failure {s['failure']['mean']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    dest.write_text(json.dumps(res, indent=2))
  pts = res["points"]
  best = max(pts, key=lambda k: (pts[k]["turns"]["mean"], int(k)))
  res["selected_iteration"] = int(best)
  res["selected_checkpoint"] = avail.get(int(best))
  res["selection_rule"] = "max validation (seed 7001, nominal) mean uninterrupted turns; ties -> later"
  dest.write_text(json.dumps(res, indent=2))
  print(f"[curve {name}] selected iteration {best}")


def _policies(specs):
  out = {}
  for s in specs:
    name, rest = s.split("=", 1)
    cond, ck = rest.split(":", 1)
    out[name] = (cond, ck)
  return out


def cmd_final(a):
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  est = load_estimator(a.estimator, map_location="cuda:0").eval() if a.estimator else None
  for name, (cond, ck) in _policies(a.policy).items():
    dest = out / f"{name}.json"
    res = json.loads(dest.read_text()) if dest.exists() else {"checkpoint": ck, "cond": cond, "results": {}}
    for preset in a.presets:
      for seed in a.seeds:
        key = f"{preset}/seed{seed}"
        if key in res["results"]:
          continue
        t0 = time.time()
        pe, extra = evaluate(cond, ck, seed, preset, EvalConfig(num_envs=a.num_envs, num_steps=600), estimator=est)
        np.savez_compressed(out / f"{name}__{preset}__seed{seed}.npz", **pe)
        res["results"][key] = {"summary": summarize(pe), **extra}
        s = res["results"][key]["summary"]
        print(f"[final {name} {key}] turns {s['turns']['mean']:.3f} failure {s['failure']['mean']:.3f} drop {s['drop']['mean']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        dest.write_text(json.dumps(res, indent=2))


def cmd_ablation(a):
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  for name, (cond, ck) in _policies(a.policy).items():
    dest = out / f"{name}.json"
    res = json.loads(dest.read_text()) if dest.exists() else {"checkpoint": ck, "cond": cond, "results": {}}
    for preset in a.presets:
      for seed in a.seeds:
        # feature mean from the normal run of the same policy / seed / preset
        key0 = f"{preset}/seed{seed}/none"
        if key0 not in res["results"]:
          pe, extra = evaluate(cond, ck, seed, preset, EvalConfig(num_envs=a.num_envs, num_steps=600), action_sensitivity=False)
          np.savez_compressed(out / f"{name}__{preset}__seed{seed}__none.npz", **pe)
          res["results"][key0] = {"summary": summarize(pe), **extra}
          dest.write_text(json.dumps(res, indent=2))
        fmean = torch.tensor(res["results"][key0]["feature_mean"], device="cuda:0")
        key_s = f"{preset}/seed{seed}/sensitivity"
        if key_s not in res["results"]:
          _, extra = evaluate(cond, ck, seed, preset, EvalConfig(num_envs=a.num_envs, num_steps=600),
                              feature_mean=fmean, action_sensitivity=True)
          res["results"][key_s] = extra.get("executed_action_change", {})
          dest.write_text(json.dumps(res, indent=2))
        for mode in ("zero", "mean", "shuffle"):
          key = f"{preset}/seed{seed}/{mode}"
          if key in res["results"]:
            continue
          t0 = time.time()
          pe, extra = evaluate(cond, ck, seed, preset, EvalConfig(num_envs=a.num_envs, num_steps=600), ablation=mode, feature_mean=fmean)
          np.savez_compressed(out / f"{name}__{preset}__seed{seed}__{mode}.npz", **pe)
          res["results"][key] = {"summary": summarize(pe)}
          s = res["results"][key]["summary"]
          print(f"[ablation {name} {key}] turns {s['turns']['mean']:.3f} failure {s['failure']['mean']:.3f} ({time.time() - t0:.0f}s)", flush=True)
          dest.write_text(json.dumps(res, indent=2))


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = p.add_subparsers(dest="cmd", required=True)
  c = sub.add_parser("curve")
  c.add_argument("--run", required=True)
  c.add_argument("--cond", required=True, choices=["A", "C", "D"])
  c.add_argument("--name")
  c.add_argument("--iterations", type=int, nargs="*")
  c.add_argument("--include-last", action="store_true")
  c.add_argument("--num-envs", type=int, default=1024)
  c.add_argument("--out", required=True)
  f = sub.add_parser("final")
  f.add_argument("--policy", action="append", required=True, help="NAME=COND:CKPT")
  f.add_argument("--presets", nargs="+", default=list(PRESETS))
  f.add_argument("--seeds", type=int, nargs="+", default=list(TEST_SEEDS))
  f.add_argument("--num-envs", type=int, default=2048)
  f.add_argument("--estimator", default=None)
  f.add_argument("--out", required=True)
  b = sub.add_parser("ablation")
  b.add_argument("--policy", action="append", required=True, help="NAME=COND:CKPT")
  b.add_argument("--presets", nargs="+", default=["nominal", "heavy_slippery"])
  b.add_argument("--seeds", type=int, nargs="+", default=list(TEST_SEEDS))
  b.add_argument("--num-envs", type=int, default=2048)
  b.add_argument("--out", required=True)
  a = p.parse_args()
  {"curve": cmd_curve, "final": cmd_final, "ablation": cmd_ablation}[a.cmd](a)


if __name__ == "__main__":
  main()
