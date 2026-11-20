#!/usr/bin/env python3
"""Phase 1 held-out comparison (seeds 6201+6202): conditions A (full distribution), B (nominal
grasp), C (robust subset), D (nominal + L2 placement perturbation) for A1-A3 and R1, plus the
robustness presets for A and B (seed 6201).  Bootstrap 95% CIs over environments; the
pre-registered decision rule (PROTOCOL.md Phase 1).  Writes results/p1_heldout.{json,md},
figures/p1_heldout.png.

    python experiments/grasp_rotation_v2/scripts/compare_p1.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

E = Path("experiments/grasp_rotation_v2")
POL = ("A1", "A2", "A3", "R1")
SEEDS = (6201, 6202)
METRICS = ("turns30", "fail30", "surv30", "yaw30", "turns60", "held60")


def metrics(path):
  tr = np.load(path)
  dt, e, code = float(tr["dt"]), tr["end_step"], tr["end_code"]
  y = tr["yaw_inc"].astype(np.float64)
  out = {}
  for steps, tag in ((600, "30"), (1200, "60")):
    ended = (e >= 0) & (e < steps)
    fail = ended & (code >= 2)
    surv = np.where(ended, e + 1, steps) * dt
    rot = y[:steps].sum(0)
    out[f"turns{tag}"] = rot / (2 * math.pi)
    out[f"fail{tag}"] = fail.astype(float)
    out[f"surv{tag}"] = surv
    out[f"yaw{tag}"] = rot / np.maximum(surv, dt)
    out[f"held{tag}"] = 1.0 - fail
  out["end_t"] = np.where((e >= 0) & (code >= 2), (e + 1) * dt, np.inf)
  return out


def load_cond(cond, preset="nominal", seeds=SEEDS, pols=POL):
  d = {}
  for p in pols:
    parts = [metrics(E / "data" / f"ho_{cond}_{p}_{preset}_s{s}.npz") for s in seeds
             if (E / "data" / f"ho_{cond}_{p}_{preset}_s{s}.npz").exists()]
    if len(parts) == len(seeds):
      d[p] = {k: np.concatenate([x[k] for x in parts]) for k in parts[0]}
  return d


def boot_diff(X, Y, metric, pols, n=2000, seed=0):
  """Mean over policies of (X - Y); CI by resampling environments within each policy."""
  rng = np.random.default_rng(seed)
  point = np.mean([X[p][metric].mean() - Y[p][metric].mean() for p in pols])
  bs = np.zeros(n)
  for p in pols:
    x, y = X[p][metric], Y[p][metric]
    bs += x[rng.integers(0, len(x), (n, len(x)))].mean(1) - y[rng.integers(0, len(y), (n, len(y)))].mean(1)
  bs /= len(pols)
  return float(point), [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def main():
  conds = {c: load_cond(c) for c in "ABCD"}
  out = {"nominal": {}, "presets": {}, "decision": {}}
  for c, d in conds.items():
    if not d:
      continue
    out["nominal"][c] = {p: {m: float(d[p][m].mean()) for m in METRICS} for p in d}
    out["nominal"][c]["mean"] = {m: float(np.mean([d[p][m].mean() for p in d])) for m in METRICS}
  A = conds["A"]
  for c in "BCD":
    if len(conds[c]) == 4 and len(A) == 4:
      out["nominal"][c]["vs_A"] = {m: dict(zip(("diff", "ci95"), boot_diff(conds[c], A, m, POL))) for m in METRICS}
      out["nominal"][c]["vs_A"]["turns30_rel"] = out["nominal"][c]["mean"]["turns30"] / out["nominal"]["A"]["mean"]["turns30"] - 1
      out["nominal"][c]["vs_A"]["all_policies_turns30_up"] = bool(all(conds[c][p]["turns30"].mean() > A[p]["turns30"].mean() for p in POL))
      out["nominal"][c]["vs_A"]["all_policies_fail30_down"] = bool(all(conds[c][p]["fail30"].mean() < A[p]["fail30"].mean() for p in POL))
  for c in "BC":
    v = out["nominal"].get(c, {}).get("vs_A")
    if v:
      out["decision"][c] = {"turns30_+15%": v["turns30_rel"] >= 0.15, "failure_lower_CI_excludes_0": v["fail30"]["ci95"][1] < 0,
                            "all_four_policies_improve": v["all_policies_turns30_up"] and v["all_policies_fail30_down"]}
      out["decision"][c]["significant"] = all(out["decision"][c].values())
  if "vs_A" in out["nominal"].get("D", {}) and "vs_A" in out["nominal"].get("B", {}):
    gB, gD = out["nominal"]["B"]["vs_A"]["turns30"]["diff"], out["nominal"]["D"]["vs_A"]["turns30"]["diff"]
    out["decision"]["D_keeps_half_of_B_turn_gain"] = bool(gD >= 0.5 * gB) if gB > 0 else False
  for pr in ("heavy_slippery", "high_delay", "obs_noise_x3"):
    a, b = load_cond("A", pr, (6201,)), load_cond("B", pr, (6201,))
    if len(a) == 4:
      out["presets"].setdefault(pr, {})["A"] = {m: float(np.mean([a[p][m].mean() for p in POL])) for m in METRICS}
      out["presets"][pr]["A_per_policy"] = {p: {m: float(a[p][m].mean()) for m in METRICS} for p in POL}
    if len(b) == 4:
      out["presets"].setdefault(pr, {})["B"] = {m: float(np.mean([b[p][m].mean() for p in POL])) for m in METRICS}
      if len(a) == 4:
        out["presets"][pr]["B_vs_A"] = {m: dict(zip(("diff", "ci95"), boot_diff(b, a, m, POL))) for m in ("turns30", "fail30", "held60")}
  (E / "results" / "p1_heldout.json").write_text(json.dumps(out, indent=1))
  L = ["# Phase 1 held-out: grasp initialisation with existing policies (seeds 6201+6202, 4096 environments per cell)", "",
       "| condition | policy | 30-s turns | 30-s failure | 30-s survival | yaw rate held | 60-s turns | P(held 60 s) |", "|---|---|---|---|---|---|---|---|"]
  for c, d in out["nominal"].items():
    for p in list(POL) + ["mean"]:
      if p in d:
        r = d[p]
        L.append(f"| {c} | {p} | {r['turns30']:.3f} | {r['fail30']:.3f} | {r['surv30']:.1f} | {r['yaw30']:.3f} | {r['turns60']:.3f} | {r['held60']:.3f} |")
  L += ["", "Differences vs A (mean over A1-A3, R1; 95% bootstrap CI):", ""]
  for c in "BCD":
    v = out["nominal"].get(c, {}).get("vs_A")
    if v:
      L.append(f"- {c}: " + "; ".join(f"{m} {v[m]['diff']:+.3f} [{v[m]['ci95'][0]:+.3f}, {v[m]['ci95'][1]:+.3f}]" for m in METRICS))
  L += ["", "Decision (PROTOCOL.md Phase 1): " + json.dumps(out["decision"])]
  (E / "results" / "p1_heldout.md").write_text("\n".join(L) + "\n")
  print("\n".join(L))
  print(json.dumps(out["presets"], indent=1)[:3000])
  figure(conds)


def figure(conds):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  names = {"A": "A: full distribution", "B": "B: nominal grasp", "C": "C: robust subset", "D": "D: nominal + placement noise"}
  cols = {"A": "#7a7a7a", "B": "#e4572e", "C": "#3b6fb6", "D": "#f2a541"}
  fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
  w = 0.2
  for i, c in enumerate("ABCD"):
    d = conds[c]
    if len(d) < 4:
      continue
    x = np.arange(4) + (i - 1.5) * w
    ax[0].bar(x, [d[p]["turns30"].mean() for p in POL], w, color=cols[c], label=names[c])
    ax[1].bar(x, [d[p]["fail30"].mean() for p in POL], w, color=cols[c])
    t = np.linspace(0, 60, 121)
    allend = np.concatenate([d[p]["end_t"] for p in POL])
    ax[2].plot(t, [(allend > s).mean() for s in t], color=cols[c], label=names[c])
  for a, ttl in zip(ax[:2], ("uninterrupted turns in 30 s", "failure rate within 30 s")):
    a.set_xticks(range(4), POL)
    a.set_title(ttl)
  ax[0].legend(fontsize=8, loc="lower right")
  ax[2].set(title="P(cube held), pooled over A1-A3, R1", xlabel="time (s)", ylim=(0.6, 1.01))
  fig.suptitle("Held-out seeds 6201+6202: effect of the starting grasp on the existing policies (no retraining)")
  fig.tight_layout()
  fig.savefig(E / "figures" / "p1_heldout.png", dpi=130)
  plt.close(fig)


if __name__ == "__main__":
  main()
