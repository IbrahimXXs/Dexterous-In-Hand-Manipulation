#!/usr/bin/env python3
"""C5 three-seed confirmation (seeds 1-3, selected checkpoints by the pre-registered validation rule).
Held-out seeds 6201+6202, same items and metrics as Phase 4 (compare_v2.py).

Seed-level inference: per-training-seed means are the unit.  C5 (3 seeds) vs A (A1-A3 = 3 seeds):
Welch t-test and 95% CI of the difference of seed means.  C5 vs R1 (one seed): one-sample t CI of
C5 seed means minus R1.  The pre-registered Phase 4 thresholds are applied to each C5 seed and to the
3-seed mean.  Writes results/c5_seeds.{json,md}, figures/c5_seeds.png.

    python experiments/grasp_rotation_v2/scripts/compare_seeds.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from compare_p1 import METRICS, metrics  # noqa: E402

E = Path("experiments/grasp_rotation_v2")
A_POL = ("A1", "A2", "A3")
C5 = ("C5", "C5s2", "C5s3")
PRESETS = ("heavy_slippery", "high_delay", "obs_noise_x3")


def mean_of(cond, pol, preset="nominal", seeds=(6201, 6202)):
  fs = [E / "data" / f"ho_{cond}_{pol}_{preset}_s{s}.npz" for s in seeds]
  if not all(f.exists() for f in fs):
    return None
  parts = [metrics(f) for f in fs]
  return {m: float(np.concatenate([x[m] for x in parts]).mean()) for m in METRICS}


def welch(x, y):
  x, y = np.asarray(x), np.asarray(y)
  d = x.mean() - y.mean()
  vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
  se = np.sqrt(vx + vy)
  df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
  t = d / se
  h = stats.t.ppf(0.975, df) * se
  return {"diff": float(d), "ci95": [float(d - h), float(d + h)], "t": float(t), "df": float(df),
          "p_two_sided": float(2 * stats.t.sf(abs(t), df))}


def one_sample(x, ref):
  x = np.asarray(x) - ref
  se = x.std(ddof=1) / np.sqrt(len(x))
  h = stats.t.ppf(0.975, len(x) - 1) * se
  t = x.mean() / se
  return {"diff": float(x.mean()), "ci95": [float(x.mean() - h), float(x.mean() + h)], "t": float(t),
          "p_two_sided": float(2 * stats.t.sf(abs(t), len(x) - 1))}


def thresholds(v, am, r1, presets):
  th = {"1_yaw_ge_A": v["yaw30"] >= am["yaw30"],
        "2_turns_ge_1.05A_and_1.10R1": v["turns30"] >= 1.05 * am["turns30"] and v["turns30"] >= 1.10 * r1["turns30"],
        "3_fail_le_midpoint": v["fail30"] <= (am["fail30"] + r1["fail30"]) / 2,
        "4_held60_ge_A": v["held60"] >= am["held60"],
        "5_robustness": all(presets.values()) and len(presets) == len(PRESETS)}
  th["promising"] = all(th.values())
  return th


def main():
  res = {"settingA": {}, "presets": {}, "settingB": {}, "inference": {}, "thresholds": {}}
  pols = list(A_POL) + ["R1"] + list(C5) + ["C5final", "C5s2final"]
  A = {p: mean_of("A", p) for p in pols}
  A = {p: v for p, v in A.items() if v is not None}
  missing = [p for p in list(A_POL) + ["R1"] + list(C5) if p not in A]
  if missing:
    sys.exit(f"missing Setting A results for {missing}")
  res["settingA"] = A
  res["settingA"]["A_mean"] = {m: float(np.mean([A[p][m] for p in A_POL])) for m in METRICS}
  res["settingA"]["C5_3seed_mean"] = {m: float(np.mean([A[p][m] for p in C5])) for m in METRICS}
  res["settingA"]["C5_3seed_sd"] = {m: float(np.std([A[p][m] for p in C5], ddof=1)) for m in METRICS}
  res["settingA"]["A_3seed_sd"] = {m: float(np.std([A[p][m] for p in A_POL], ddof=1)) for m in METRICS}
  for m in METRICS:
    res["inference"][m] = {"C5_vs_A_welch": welch([A[p][m] for p in C5], [A[p][m] for p in A_POL]),
                           "C5_vs_R1_one_sample": one_sample([A[p][m] for p in C5], A["R1"][m])}
  for pr in PRESETS:
    T = {p: mean_of("A", p, pr, (6201,)) for p in list(A_POL) + ["R1"] + list(C5)}
    T = {p: v for p, v in T.items() if v is not None}
    if all(p in T for p in A_POL):
      T["A_mean"] = {m: float(np.mean([T[p][m] for p in A_POL])) for m in METRICS}
    res["presets"][pr] = T
  am, r1 = res["settingA"]["A_mean"], A["R1"]
  for p in list(C5) + ["C5_3seed_mean"]:
    v = res["settingA"][p]
    if p == "C5_3seed_mean":
      rob = {pr: float(np.mean([res["presets"][pr][q]["turns30"] for q in C5])) >= 0.9 * res["presets"][pr]["A_mean"]["turns30"]
             for pr in PRESETS}
    else:
      rob = {pr: res["presets"][pr][p]["turns30"] >= 0.9 * res["presets"][pr]["A_mean"]["turns30"] for pr in PRESETS}
    res["thresholds"][p] = thresholds(v, am, r1, rob)
  for cond, nm in (("B", "nominal_grasp_49"), ("C", "robust_subset")):
    T = {p: mean_of(cond, p) for p in list(A_POL) + ["R1"] + list(C5)}
    T = {p: v for p, v in T.items() if v is not None}
    if all(p in T for p in A_POL):
      T["A_mean"] = {m: float(np.mean([T[p][m] for p in A_POL])) for m in METRICS}
    if all(p in T for p in C5):
      T["C5_3seed_mean"] = {m: float(np.mean([T[p][m] for p in C5])) for m in METRICS}
      T["C5_vs_A_welch_turns30"] = welch([T[p]["turns30"] for p in C5], [T[p]["turns30"] for p in A_POL])
    res["settingB"][nm] = T
  N = {p: mean_of("N", p, "nominal", (6201,)) for p in list(A_POL) + ["R1"] + list(C5)}
  res["supplementary_no_pose_rule"] = {p: v for p, v in N.items() if v is not None}
  (E / "results" / "c5_seeds.json").write_text(json.dumps(res, indent=1, default=bool))
  md(res)
  figure(res)


def row(p, r):
  return f"| {p} | {r['turns30']:.3f} | {r['fail30']:.3f} | {r['yaw30']:.3f} | {r['turns60']:.3f} | {r['held60']:.3f} |"


def md(res):
  H = "| policy | 30-s turns | 30-s failure | yaw rate held | 60-s turns | P(held 60 s) |\n|---|---|---|---|---|---|"
  L = ["# C5 three-seed confirmation (generated by scripts/compare_seeds.py)", "",
       "Held-out seeds 6201+6202, 4096 environments per policy; selected checkpoints (max validation turns, seed 7001).", "",
       "## Setting A (main): full original grasp distribution", "", H]
  A = res["settingA"]
  for p in ("A1", "A2", "A3", "A_mean", "R1", "C5", "C5s2", "C5s3", "C5_3seed_mean", "C5final", "C5s2final"):
    if p in A:
      L.append(row(p, A[p]))
  L += ["", f"Seed SD (C5): turns30 {A['C5_3seed_sd']['turns30']:.3f}, fail30 {A['C5_3seed_sd']['fail30']:.3f}; "
            f"(A1-A3): turns30 {A['A_3seed_sd']['turns30']:.3f}, fail30 {A['A_3seed_sd']['fail30']:.3f}", "",
        "## Seed-level inference (unit = training seed)", "",
        "| metric | C5 − A (Welch, 3 vs 3 seeds) | p | C5 − R1 (one-sample, 3 seeds vs 1) | p |", "|---|---|---|---|---|"]
  for m, v in res["inference"].items():
    a, r = v["C5_vs_A_welch"], v["C5_vs_R1_one_sample"]
    L.append(f"| {m} | {a['diff']:+.3f} [{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}] | {a['p_two_sided']:.4f} | "
             f"{r['diff']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['p_two_sided']:.4f} |")
  L += ["", "## Robustness presets (seed 6201), turns30 / failure", "", "| preset | A mean | R1 | C5 | C5s2 | C5s3 |", "|---|---|---|---|---|---|"]
  for pr, T in res["presets"].items():
    L.append(f"| {pr} | " + " | ".join(f"{T[p]['turns30']:.3f} / {T[p]['fail30']:.3f}" if p in T else "—" for p in ("A_mean", "R1", "C5", "C5s2", "C5s3")) + " |")
  for nm, T in res["settingB"].items():
    L += ["", f"## Setting B: {nm.replace('_', ' ')}", "", H]
    for p in ("A1", "A2", "A3", "A_mean", "R1", "C5", "C5s2", "C5s3", "C5_3seed_mean"):
      if p in T:
        L.append(row(p, T[p]))
    if "C5_vs_A_welch_turns30" in T:
      w = T["C5_vs_A_welch_turns30"]
      L += ["", f"C5 − A turns30 (seed level): {w['diff']:+.3f} [{w['ci95'][0]:+.3f}, {w['ci95'][1]:+.3f}], p {w['p_two_sided']:.4f}"]
  L += ["", "## Supplementary: pose-deviation rule off (seed 6201)", "", H]
  for p, r in res["supplementary_no_pose_rule"].items():
    L.append(row(p, r))
  L += ["", "## Pre-registered Phase 4 thresholds per seed and for the 3-seed mean", "",
        "| | 1 yaw ≥ A | 2 turns ≥ 1.05A & 1.10R1 | 3 fail ≤ midpoint | 4 held60 ≥ A | 5 presets | promising |", "|---|---|---|---|---|---|---|"]
  for p, th in res["thresholds"].items():
    L.append(f"| {p} | " + " | ".join("✓" if th[k] else "✗" for k in ("1_yaw_ge_A", "2_turns_ge_1.05A_and_1.10R1", "3_fail_le_midpoint", "4_held60_ge_A", "5_robustness", "promising")) + " |")
  (E / "results" / "c5_seeds.md").write_text("\n".join(L) + "\n")
  print("\n".join(L))


def figure(res):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
  cur = {}
  for r, f in (("A1", "../hwlimit_ac/results/curves/A1.json"), ("A2", "../hwlimit_ac/results/curves/A2.json"),
               ("A3", "../hwlimit_ac/results/curves/A3.json"), ("R1", "../retention_v1/results/curves/R1.json"),
               ("C5_1", "results/curves/C5_1.json"), ("C5_2", "results/curves/C5_2.json"), ("C5_3", "results/curves/C5_3.json")):
    p = E / f
    if p.exists():
      pts = json.loads(p.read_text())["points"]
      its = sorted(int(k) for k in pts)
      cur[r] = (its, [pts[str(i)]["turns"]["mean"] for i in its], [pts[str(i)]["failure"]["mean"] for i in its])
  for r, (its, v, f) in cur.items():
    c = "#9a9a9a" if r.startswith("A") else "#3b6fb6" if r == "R1" else "#2a9d8f"
    ax[0].plot(its, v, color=c, lw=1.8 if r.startswith("C5") else 1.1, label={"A1": "A1-A3", "R1": "R1", "C5_1": "C5 seeds 1-3"}.get(r))
    ax[1].plot(its, f, color=c, lw=1.8 if r.startswith("C5") else 1.1)
  ax[0].set(title="validation turns (30 s), seed 7001", xlabel="PPO update")
  ax[1].set(title="validation failure rate", xlabel="PPO update")
  ax[0].legend(fontsize=8)
  A = res["settingA"]
  groups = (("A", A_POL, "#9a9a9a"), ("R1", ("R1",), "#3b6fb6"), ("C5", C5, "#2a9d8f"))
  for a, m, t in ((ax[2], "turns30", "held-out 30-s turns (full distribution)"), (ax[3], "fail30", "held-out 30-s failure")):
    for i, (g, ps, c) in enumerate(groups):
      vals = [A[p][m] for p in ps]
      a.bar(i, np.mean(vals), color=c, alpha=0.6)
      a.scatter([i] * len(vals), vals, color="k", s=14, zorder=3)
    a.set_xticks(range(3), [g for g, _, _ in groups])
    a.set_title(t)
  fig.suptitle("C5 confirmation: three independent training seeds (dots = training seeds)")
  fig.tight_layout()
  fig.savefig(E / "figures" / "c5_seeds.png", dpi=130)
  plt.close(fig)


if __name__ == "__main__":
  main()
