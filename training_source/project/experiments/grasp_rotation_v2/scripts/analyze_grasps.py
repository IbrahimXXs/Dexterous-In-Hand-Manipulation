#!/usr/bin/env python3
"""Phase 1 analysis: per-grasp performance on the development rollouts (seeds 6101, 6102),
grasp features, the pre-registered ranking (PROTOCOL.md Phase 1), the robust subset and
figures.  Writes results/grasp_table.csv, results/grasp_ranking.json, figures/p1_*.png.

    python experiments/grasp_rotation_v2/scripts/analyze_grasps.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

E = Path("experiments/grasp_rotation_v2")
POL = ("A1", "A2", "A3", "R1")
SEEDS = (6101, 6102)
NG = 239
KIT_START = np.array([0.1, -0.4, 1.3, 0.0, 0.1, 0.0, 1.3, 0.0, 0.1, 0.4, 1.3, 0.0, 1.45, 1.5, 0.579, 1.37])
JOINTS = ["if_mcp", "if_rot", "if_pip", "if_dip", "mf_mcp", "mf_rot", "mf_pip", "mf_dip", "rf_mcp", "rf_rot", "rf_pip",
          "rf_dip", "th_cmc", "th_axl", "th_mcp", "th_ipl"]


def load(pol, seed):
  return dict(np.load(E / "data" / f"p1dev_{pol}_s{seed}.npz"))


def per_env(tr, steps):
  dt = float(tr["dt"])
  e = tr["end_step"]
  ended = (e >= 0) & (e < steps)
  fail = ended & (tr["end_code"] >= 2)
  surv = np.where(ended, e + 1, steps) * dt
  rot = tr["yaw_inc"][:steps].astype(np.float64).sum(0)
  return {"turns": rot / (2 * math.pi), "fail": fail.astype(float), "surv": surv, "yaw": rot / np.maximum(surv, dt),
          "fail_t": np.where(fail, (e + 1) * dt, np.nan), "tilt_fail": (fail & (tr["end_code"] == 4)).astype(float)}


def cube_tilt_in_palm(q):
  """Angle (rad) between the palm normal (palm-frame z) and the nearest cube face normal."""
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  # third row of the rotation matrix = palm-z expressed in cube axes
  r = np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)
  return np.arccos(np.clip(np.abs(r).max(-1), 0, 1))


def grasp_stats():
  rows = {g: {} for g in range(NG)}
  feats = None
  env_rows = []
  for pol in POL + ("hold",):
    for seed in SEEDS:
      tr = load(pol, seed)
      cid = tr["cache_id"]
      m30, m60 = per_env(tr, 600), per_env(tr, 1200)
      for g in range(NG):
        k = cid == g
        r = rows[g].setdefault(pol, {"T30": [], "T60": [], "P30": [], "P60": [], "yaw60": [], "fail_t": [], "seed": []})
        r["T30"] += m30["turns"][k].tolist()
        r["T60"] += m60["turns"][k].tolist()
        r["P30"] += (1 - m30["fail"][k]).tolist()
        r["P60"] += (1 - m60["fail"][k]).tolist()
        r["yaw60"] += m60["yaw"][k].tolist()
        r["fail_t"] += m60["fail_t"][k].tolist()
        r["seed"] += [seed] * int(k.sum())
      if feats is None and pol == "A1":
        q = tr["init_quat_palm"]
        f = {"pos_x": tr["init_pos_palm"][:, 0], "pos_y": tr["init_pos_palm"][:, 1], "pos_z": tr["init_pos_palm"][:, 2],
             "tilt_palm": cube_tilt_in_palm(q), "n_contact": tr["init_contact"].sum(-1),
             "min_headroom": tr["init_headroom"].min(-1), "n_at_limit": (tr["init_headroom"] < 1e-3).sum(-1),
             "tip_dist_mean": np.linalg.norm(tr["init_tip_rel_palm"], axis=-1).mean(-1)}
        for i, c in enumerate(("if", "mf", "rf", "th")):
          f[f"contact_{c}"] = tr["init_contact"][:, i].astype(float)
          f[f"tipdist_{c}"] = np.linalg.norm(tr["init_tip_rel_palm"][:, i], axis=-1)
        feats = {k: np.array([np.mean(v[cid == g]) for g in range(NG)]) for k, v in f.items()}
        feats["init_command"] = np.stack([tr["init_command"][cid == g].mean(0) for g in range(NG)])
        feats["init_headroom"] = np.stack([tr["init_headroom"][cid == g].mean(0) for g in range(NG)])
  return rows, feats


def main():
  rows, feats = grasp_stats()
  agg = {}
  for g in range(NG):
    a = {}
    for pol in POL + ("hold",):
      r = rows[g][pol]
      a[pol] = {k: float(np.nanmean(v)) if len(v) and not np.all(np.isnan(v)) else float("nan")
                for k, v in r.items() if k not in ("seed",)}
      a[pol]["n"] = len(r["T60"])
      sd = np.array(r["seed"])
      for s in SEEDS:
        a[pol][f"T60_s{s}"] = float(np.mean(np.array(r["T60"])[sd == s]))
    agg[g] = a
  S = np.array([np.mean([agg[g][p]["T60"] for p in POL]) for g in range(NG)])
  S_half = {s: np.array([np.mean([agg[g][p][f"T60_s{s}"] for p in POL]) for g in range(NG)]) for s in SEEDS}
  split_r = float(np.corrcoef(S_half[6101], S_half[6102])[0, 1])
  P60 = np.array([np.mean([agg[g][p]["P60"] for p in POL]) for g in range(NG)])
  P30 = np.array([np.mean([agg[g][p]["P30"] for p in POL]) for g in range(NG)])
  Y = np.array([np.mean([agg[g][p]["yaw60"] for p in POL]) for g in range(NG)])
  T60p = {p: np.array([agg[g][p]["T60"] for g in range(NG)]) for p in POL}
  T30 = np.array([np.mean([agg[g][p]["T30"] for p in POL]) for g in range(NG)])
  e1 = P60 >= 0.80
  e2 = Y >= np.median(Y)
  e3 = np.all([T60p[p] >= np.median(T60p[p]) for p in POL], axis=0)
  elig = e1 & e2 & e3
  order = [g for g in np.argsort(-S) if elig[g]]
  cmd = feats["init_command"]
  subset = []
  for g in order:
    if all(np.sqrt(((cmd[g] - cmd[h]) ** 2).mean()) >= 0.05 for h in subset):
      subset.append(int(g))
    if len(subset) == 10:
      break
  kit_dist = np.sqrt(((cmd - KIT_START) ** 2).mean(-1))
  # feature association with S (Spearman)
  from scipy.stats import spearmanr

  scal = {k: v for k, v in feats.items() if v.ndim == 1}
  scal["kit_start_dist"] = kit_dist
  for j, nm in enumerate(JOINTS):
    scal[f"cmd_{nm}"] = cmd[:, j]
    scal[f"headroom_{nm}"] = feats["init_headroom"][:, j]
  hold_ret = np.array([agg[g]["hold"]["P60"] for g in range(NG)])
  scal["hold_P60"] = hold_ret
  assoc = {}
  for k, v in scal.items():
    if np.std(v) > 0:
      rho, pv = spearmanr(v, S)
      rho_p, _ = spearmanr(v, P60)
      rho_y, _ = spearmanr(v, Y)
      assoc[k] = {"rho_S": float(rho), "p_S": float(pv), "rho_P60": float(rho_p), "rho_yaw": float(rho_y)}
  # cross-validated ridge on standardised scalar features (how much of S do grasp features explain?)
  X = np.stack([scal[k] for k in sorted(scal) if np.std(scal[k]) > 0], 1)
  X = (X - X.mean(0)) / X.std(0)
  rng = np.random.default_rng(0)
  folds = rng.permutation(NG) % 5
  pred = np.zeros(NG)
  for f in range(5):
    tr_, te = folds != f, folds == f
    A = X[tr_]
    w = np.linalg.solve(A.T @ A + 10.0 * np.eye(A.shape[1]), A.T @ (S[tr_] - S[tr_].mean()))
    pred[te] = X[te] @ w + S[tr_].mean()
  cv_r2 = float(1 - ((S - pred) ** 2).sum() / ((S - S.mean()) ** 2).sum())
  # policy agreement per grasp
  pol_corr = {f"{p}-{q}": float(np.corrcoef(T60p[p], T60p[q])[0, 1]) for i, p in enumerate(POL) for q in POL[i + 1:]}
  out = {"split_half_r_S": split_r, "n_eligible": int(elig.sum()), "eligibility_counts": {"P60>=0.8": int(e1.sum()),
         "yaw>=median": int(e2.sum()), "all_policies_T60>=median": int(e3.sum())},
         "median_yaw_rate": float(np.median(Y)), "ranking_top20": [int(g) for g in order[:20]],
         "robust_subset_C": subset, "subset_S": [float(S[g]) for g in subset],
         "all_grasps_mean_S": float(S.mean()), "all_grasps_mean_T30": float(T30.mean()), "all_grasps_mean_P60": float(P60.mean()),
         "policy_T60_corr_across_grasps": pol_corr, "feature_assoc_spearman": dict(sorted(assoc.items(), key=lambda kv: -abs(kv[1]["rho_S"]))),
         "ridge_cv_r2_S_from_features": cv_r2,
         "S_quantiles": np.quantile(S, [0, .1, .25, .5, .75, .9, 1]).round(3).tolist(),
         "kit_start_dist_rms_rad": {"min": float(kit_dist.min()), "median": float(np.median(kit_dist)), "subset": [float(kit_dist[g]) for g in subset]}}
  (E / "results" / "grasp_ranking.json").write_text(json.dumps(out, indent=1))
  hdr = ["grasp", "S_T60_mean", "T30_mean", "P30_mean", "P60_mean", "yaw60_mean", "eligible", "rank"] + \
        [f"{p}_{m}" for p in POL + ("hold",) for m in ("T30", "T60", "P60", "yaw60", "fail_t")] + sorted(k for k in scal)
  rank = {g: i + 1 for i, g in enumerate(order)}
  lines = [",".join(hdr)]
  for g in range(NG):
    v = [g, S[g], T30[g], P30[g], P60[g], Y[g], int(elig[g]), rank.get(g, "")]
    v += [agg[g][p][m] for p in POL + ("hold",) for m in ("T30", "T60", "P60", "yaw60", "fail_t")]
    v += [scal[k][g] for k in sorted(scal)]
    lines.append(",".join(f"{x:.5g}" if isinstance(x, float) else str(x) for x in v))
  (E / "results" / "grasp_table.csv").write_text("\n".join(lines) + "\n")
  np.savez(E / "results" / "grasp_arrays.npz", S=S, S_6101=S_half[6101], S_6102=S_half[6102], P60=P60, P30=P30, Y=Y, T30=T30,
           eligible=elig, kit_dist=kit_dist, **{f"T60_{p}": T60p[p] for p in POL}, **{f"f_{k}": v for k, v in scal.items()})
  print(json.dumps({k: v for k, v in out.items() if k != "feature_assoc_spearman"}, indent=1))
  print("top feature associations:")
  for k, v in list(out["feature_assoc_spearman"].items())[:15]:
    print(f"  {k:24s} rho_S {v['rho_S']:+.2f} rho_P60 {v['rho_P60']:+.2f} rho_yaw {v['rho_yaw']:+.2f}")
  figures(S, S_half, P60, Y, T60p, scal, elig, subset)


def figures(S, S_half, P60, Y, T60p, scal, elig, subset):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  F = E / "figures"
  F.mkdir(exist_ok=True)
  fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
  ax[0].scatter(S_half[6101], S_half[6102], s=10, c="#3b6fb6")
  lim = [0, max(S_half[6101].max(), S_half[6102].max()) * 1.05]
  ax[0].plot(lim, lim, "k:", lw=0.8)
  ax[0].set(xlabel="grasp score S, seed 6101", ylabel="grasp score S, seed 6102",
            title=f"split-half reliability r = {np.corrcoef(S_half[6101], S_half[6102])[0, 1]:.2f}")
  Ta = np.mean([T60p[p] for p in ("A1", "A2", "A3")], 0)
  ax[1].scatter(Ta, T60p["R1"], s=10, c="#3b6fb6")
  ax[1].plot([0, Ta.max()], [0, Ta.max()], "k:", lw=0.8)
  ax[1].set(xlabel="60-s turns, mean of A1-A3", ylabel="60-s turns, R1", title=f"same grasps are hard for A and R1 (r = {np.corrcoef(Ta, T60p['R1'])[0, 1]:.2f})")
  sc = ax[2].scatter(P60, Y, c=S, s=12, cmap="viridis")
  ax[2].scatter(P60[subset], Y[subset], s=60, facecolors="none", edgecolors="#e4572e", label="robust subset")
  ax[2].axvline(0.8, color="k", ls=":", lw=0.8)
  ax[2].axhline(np.median(Y), color="k", ls=":", lw=0.8)
  ax[2].set(xlabel="P(held 60 s), mean of 4 policies", ylabel="yaw rate while held (rad/s)", title="retention vs rotation speed per grasp")
  ax[2].legend(loc="lower left")
  fig.colorbar(sc, ax=ax[2], label="S (60-s turns)")
  fig.tight_layout()
  fig.savefig(F / "p1_grasp_reliability.png", dpi=130)
  plt.close(fig)
  keys = ["pos_x", "pos_y", "pos_z", "tilt_palm", "n_contact", "kit_start_dist"]
  fig, ax = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.4), sharey=True)
  for a, k in zip(ax, keys):
    a.scatter(scal[k], S, s=8, c=np.where(elig, "#e4572e", "#3b6fb6"))
    a.set(xlabel=k)
  ax[0].set(ylabel="S (60-s turns)")
  fig.suptitle("grasp score vs initial-grasp features (orange = eligible)")
  fig.tight_layout()
  fig.savefig(F / "p1_grasp_features.png", dpi=130)
  plt.close(fig)


if __name__ == "__main__":
  main()
