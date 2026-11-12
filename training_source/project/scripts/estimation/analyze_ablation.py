#!/usr/bin/env python3
"""Paired analysis of estimator-feature ablations (evaluate_policies.py --ablation).

All runs share seeds, presets and per-environment randomisation, so each
environment is compared with itself: difference = metric(ablation) -
metric(baseline) per env, mean and 95% CI over envs.  The GPU simulation is
not bit-reproducible, so a *repeat* of the baseline run (``--null``) gives the
run-to-run noise floor that any real effect has to exceed.

    python scripts/estimation/analyze_ablation.py \
        --run none=data/eval/diagnostics/ablation/none --run zero=... \
        --null data/eval/policies/C_estimator --baseline none \
        --output data/eval/diagnostics/ablation/summary.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

METRICS = ("turns", "failure", "survival_s", "reward_sum", "drop", "drift_max_mm", "tilt_max_deg")


def load(d: str) -> dict[tuple[str, str], dict[str, np.ndarray]]:
  out = {}
  for f in glob.glob(str(Path(d) / "*.npz")):
    _, preset, seed = Path(f).stem.split("__")
    out[(preset, seed)] = dict(np.load(f))
  return out


def paired(a: dict, b: dict, keys) -> dict:
  res = {}
  for m in METRICS:
    x = np.concatenate([a[k][m] for k in keys]).astype(float)
    y = np.concatenate([b[k][m] for k in keys]).astype(float)
    d = x - y
    res[m] = {"mean": float(x.mean()), "baseline_mean": float(y.mean()), "diff": float(d.mean()),
              "diff_ci95": float(1.96 * d.std(ddof=1) / math.sqrt(len(d))), "n": int(len(d))}
  return res


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--run", action="append", required=True, help="NAME=DIR")
  p.add_argument("--baseline", default="none")
  p.add_argument("--null", default=None, help="DIR with a repeat of the baseline run")
  p.add_argument("--output", required=True)
  a = p.parse_args()
  runs = {k: load(v) for k, v in (r.split("=", 1) for r in a.run)}
  if a.null:
    runs["_baseline_repeat"] = load(a.null)
  base = runs[a.baseline]
  presets = sorted({pr for pr, _ in base})
  report = {}
  for pr in presets:
    report[pr] = {}
    for name, r in runs.items():
      if name == a.baseline:
        continue
      keys = [k for k in base if k[0] == pr and k in r]
      if keys:
        report[pr][name] = paired(r, base, keys)
  Path(a.output).write_text(json.dumps(report, indent=2))
  for pr, d in report.items():
    print(f"\n## {pr}  (paired difference vs '{a.baseline}', mean ± 95% CI over envs)")
    print("| run | " + " | ".join(METRICS) + " |")
    print("|---|" + "---|" * len(METRICS))
    for name, m in d.items():
      print(f"| {name} | " + " | ".join(f"{m[k]['diff']:+.3f} ± {m[k]['diff_ci95']:.3f}" for k in METRICS) + " |")
    any_m = next(iter(d.values()))
    print("| (baseline level) | " + " | ".join(f"{any_m[k]['baseline_mean']:.3f}" for k in METRICS) + " |")


if __name__ == "__main__":
  main()
