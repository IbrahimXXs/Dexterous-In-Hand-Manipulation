#!/usr/bin/env python3
"""C5-HW-V2: per-evaluation-seed breakdown and descriptive CIs (added after the decision; descriptive only).

Effort actuator, nominal.  For settings K, A, B and C: C5HW2_1, the C5 3-seed mean (per-environment mean of
C5_1-C5_3) and EFFORT_1, per evaluation seed and pooled, with paired-bootstrap CIs (compare_v2.boot_diff /
boot_ratio, 10,000 resamples within each seed) of C5HW2_1 - C5 mean and C5HW2_1 - EFFORT_1 for 30-s turns and
30-s failure, and the termination reasons of every held-out C5HW2_1 rollout.  Writes results/v2_per_eval_seed.json.

    python experiments/c5_hw_v2/scripts/per_eval_seed.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

E = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("compare_v2", E / "scripts" / "compare_v2.py")
CV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CV)

SEEDS = (6201, 6202)
NEW, EFF, C5 = "C5HW2_1", "EFFORT_1", ("C5_1", "C5_2", "C5_3")
FIELDS = ("turns30", "failure30", "jam60")


def main() -> None:
  rng = np.random.default_rng(20261128)
  out = {"label": "descriptive (not a criterion); single-seed exploratory; effort actuator, nominal", "settings": {}}
  for st in ("K", "A", "B", "C"):
    new = {f: [CV.per_env(CV.key(st, NEW, "effort", "nominal", s))[f] for s in SEEDS] for f in FIELDS}
    eff = {f: [CV.per_env(CV.key(st, EFF, "effort", "nominal", s))[f] for s in SEEDS] for f in FIELDS}
    c5 = {f: CV.mean_of(C5, st, "effort", "nominal", SEEDS, f) for f in FIELDS}
    row = {"per_seed": {}, "pooled": {}}
    for i, s in enumerate(SEEDS):
      row["per_seed"][str(s)] = {f: {"C5HW2_1": float(new[f][i].mean()), "C5_mean": float(c5[f][i].mean()),
                                     "EFFORT_1": float(eff[f][i].mean())} for f in FIELDS}
      row["per_seed"][str(s)]["turns30_ratio_vs_C5_mean"] = float(new["turns30"][i].mean() / c5["turns30"][i].mean())
    for f in FIELDS:
      row["pooled"][f] = {"C5HW2_1": float(np.concatenate(new[f]).mean()), "C5_mean": float(np.concatenate(c5[f]).mean()),
                          "EFFORT_1": float(np.concatenate(eff[f]).mean())}
    for f in ("turns30", "failure30"):
      row["pooled"][f]["diff_vs_C5_mean"] = CV.boot_diff(new[f], c5[f], rng)
      row["pooled"][f]["diff_vs_EFFORT_1"] = CV.boot_diff(new[f], eff[f], rng)
    row["pooled"]["turns30_ratio_vs_C5_mean"] = CV.boot_ratio(new["turns30"], c5["turns30"], rng)
    row["pooled"]["turns30_ratio_vs_EFFORT_1"] = CV.boot_ratio(new["turns30"], eff["turns30"], rng)
    out["settings"][st] = row
  # termination reasons over every held-out C5HW2_1 rollout (all settings, actuators, presets)
  codes = {1: "time_out", 2: "cube_fell", 3: "cube_too_fast", 4: "cube_pose_deviation", 5: "nan"}
  cnt, n_env, n_items = {v: 0 for v in codes.values()}, 0, 0
  for f in sorted((E / "data" / "items").glob(f"*_{NEW}_*.npz")):
    tr = np.load(f)
    ended = (tr["end_step"] >= 0) & (tr["end_step"] < int(tr["T"]))
    for c, name in codes.items():
      cnt[name] += int((tr["end_code"][ended] == c).sum())
    n_env += len(tr["end_step"]); n_items += 1
  out["C5HW2_1_heldout_terminations"] = {"items": n_items, "environments": n_env, "early_terminations_by_reason": cnt}
  (E / "results" / "v2_per_eval_seed.json").write_text(json.dumps(out, indent=1) + "\n")
  for st, r in out["settings"].items():
    p = r["pooled"]
    print(st, "turns", {k: round(v, 3) for k, v in p["turns30"].items() if isinstance(v, float)},
          "dT_C5", p["turns30"]["diff_vs_C5_mean"], "ratioC5", p["turns30_ratio_vs_C5_mean"]["ratio"],
          "fail", {k: round(v, 3) for k, v in p["failure30"].items() if isinstance(v, float)}, "dF_C5", p["failure30"]["diff_vs_C5_mean"],
          "per-seed ratio", [round(r["per_seed"][str(s)]["turns30_ratio_vs_C5_mean"], 3) for s in SEEDS])


if __name__ == "__main__":
  main()
