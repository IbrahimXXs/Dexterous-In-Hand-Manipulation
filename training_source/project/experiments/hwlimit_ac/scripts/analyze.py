#!/usr/bin/env python3
"""Phase 5 analysis: tables, uncertainty, figures (PROTOCOL.md Section 5).

Reads results/test/*.npz|json, results/curves/*.json, results/convergence.json and writes
results/test_summary.json, results/tables.md, figures/learning_curves.png,
figures/test_comparison.png.

Statistics for C - A (per metric, preset, checkpoint kind):
  * per-seed means pool both test seeds (2 x 1024 environments);
  * two-level bootstrap (10 000 reps): resample the 3 training seeds of each condition
    with replacement, then environments within each resampled seed;
  * Welch t-interval on the 3 vs 3 seed means;
  * exact one-sided permutation p (C > A) over the 20 splits of the 6 seed means
    (minimum attainable 0.05).
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

E = Path("experiments/hwlimit_ac")
PRESETS = ("nominal", "heavy_slippery", "light_grippy", "high_delay", "weak_motors", "obs_noise_x3")
SEEDS = (1, 2, 3)
METRICS = ["turns", "turns_with_resets", "mean_yaw_rate", "survival_s", "completed", "failure", "drop", "pose_failure",
           "too_fast", "rotating_fraction", "drift_mean_mm", "tilt_mean_deg", "tilt_max_deg", "cmd_at_limit_frac",
           "any_joint_at_limit_frac", "kit_clip_frac", "max_cmd_step_rad", "state_outside_frac", "action_change",
           "exec_saturation"]
COL = {"A": "#2a78d6", "C": "#eb6834"}  # reference palette, categorical slots 1-2


def ckname(cond, seed, kind):
  m = [l.split(" | ")[0] for l in (E / "checkpoints" / "MANIFEST.txt").read_text().splitlines()
       if l.startswith(f"{cond}_seed{seed}_{kind}_")]
  return m[0].removesuffix(".pt")


def load_env(cond, seed, kind, preset):
  arrs = []
  for ts in (5001, 5002):
    f = E / "results" / "test" / f"{ckname(cond, seed, kind)}__{preset}__seed{ts}.npz"
    if not f.exists():
      return None
    arrs.append(dict(np.load(f)))
  return {k: np.concatenate([a[k] for a in arrs]) for k in arrs[0]}


def stats(A: list[np.ndarray], C: list[np.ndarray], rng) -> dict:
  ma, mc = np.array([a.mean() for a in A]), np.array([c.mean() for c in C])
  diff = mc.mean() - ma.mean()
  boots = np.empty(10000)
  for b in range(len(boots)):
    sa = rng.integers(0, 3, 3)
    sc = rng.integers(0, 3, 3)
    va = np.mean([A[i][rng.integers(0, len(A[i]), len(A[i]))].mean() for i in sa])
    vc = np.mean([C[i][rng.integers(0, len(C[i]), len(C[i]))].mean() for i in sc])
    boots[b] = vc - va
  va_, vc_ = ma.var(ddof=1) / 3, mc.var(ddof=1) / 3
  se = math.sqrt(va_ + vc_)
  df = (va_ + vc_) ** 2 / ((va_ ** 2 + vc_ ** 2) / 2) if se > 0 else 4
  from scipy.stats import t as student_t

  t = float(student_t.ppf(0.975, df))
  allm = np.concatenate([ma, mc])
  perm = [allm[list(c)].mean() - allm[[i for i in range(6) if i not in c]].mean() for c in itertools.combinations(range(6), 3)]
  p = float(np.mean(np.array(perm) >= diff - 1e-12))
  return {"A_seed_means": ma.tolist(), "C_seed_means": mc.tolist(), "A_mean": float(ma.mean()), "C_mean": float(mc.mean()),
          "diff_C_minus_A": float(diff), "boot_ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
          "welch_ci95": [float(diff - t * se), float(diff + t * se)], "welch_df": float(df),
          "perm_p_one_sided_C_gt_A": p}


def main():
  rng = np.random.default_rng(0)
  out = {"kinds": {}}
  for kind in ("selected", "final"):
    out["kinds"][kind] = {}
    for preset in PRESETS:
      data = {c: [load_env(c, s, kind, preset) for s in SEEDS] for c in "AC"}
      if any(d is None for c in "AC" for d in data[c]):
        out["kinds"][kind][preset] = "incomplete"
        continue
      res = {}
      for m in METRICS:
        A = [d[m].astype(np.float64) for d in data["A"]]
        C = [d[m].astype(np.float64) for d in data["C"]]
        if m == "max_cmd_step_rad":
          res[m] = {"A_max": float(max(a.max() for a in A)), "C_max": float(max(c.max() for c in C))}
          continue
        res[m] = stats(A, C, rng) if m in ("turns", "turns_with_resets", "mean_yaw_rate", "survival_s", "failure") else {
          "A_seed_means": [float(a.mean()) for a in A], "C_seed_means": [float(c.mean()) for c in C],
          "A_mean": float(np.mean([a.mean() for a in A])), "C_mean": float(np.mean([c.mean() for c in C]))}
      out["kinds"][kind][preset] = res
  # estimator audit (nominal rollouts)
  audit = {}
  for c in "AC":
    rows = []
    for s in SEEDS:
      for ts in (5001, 5002):
        j = json.loads((E / "results" / "test" / f"{ckname(c, s, 'selected')}__nominal__seed{ts}.json").read_text())
        el = j["extra"].get("estimator_closed_loop")
        if el:
          fr = np.array(list(j["extra"]["feature_rmse"].values()))
          rows.append({"pos_rmse_mm": el["pos_rmse_mm"], "rot_err_sym_mean_deg": el["rot_err_sym_mean_deg"],
                       "lin_vel_rmse_mps": el["lin_vel_rmse_mps"], "ang_vel_rmse_radps": el["ang_vel_rmse_radps"],
                       "feature_rmse_all": float(np.sqrt((fr ** 2).mean())),
                       "contact_f1_mean": float(np.mean([el[f"contact_{f}_f1"] for f in ("if", "mf", "rf", "th")])),
                       "input_mean_abs_z": j["extra"]["input_shift"]["mean_abs_z"]})
    audit[f"{c}_policies"] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else None
  ref = json.loads((E / "results" / "estimator_validation.json").read_text())["aggregate"]
  audit["phase2_reference_original_env"] = ref["original"]
  audit["phase2_old_policies_hwlimit_env"] = ref["hwlimit"]
  out["estimator_audit"] = audit
  conv = json.loads((E / "results" / "convergence.json").read_text())
  out["convergence"] = {r: {k: v.get(k) for k in ("status", "selected_iteration", "selected_V", "final_V",
                                                  "learning_speed_mean_V_100_3000", "first_iteration_V_ge_0.5")}
                        for r, v in conv["runs"].items()}
  (E / "results" / "test_summary.json").write_text(json.dumps(out, indent=1))
  write_tables(out)
  figures(conv, out)


def f3(x):
  return f"{x:.3f}"


def write_tables(out):
  L = ["# Hardware-limit A vs C: results (generated by scripts/analyze.py)", ""]
  L += ["## Convergence and selection (validation seed 7001, nominal)", "",
        "| run | status | selected | V selected | V final (3000) | mean V 100-3000 | first V>=0.5 |", "|---|---|---|---|---|---|---|"]
  for r, v in out["convergence"].items():
    L.append(f"| {r} | {v['status']} | {v['selected_iteration']} | {f3(v['selected_V'])} | {f3(v['final_V'])} | "
             f"{f3(v['learning_speed_mean_V_100_3000'])} | {v['first_iteration_V_ge_0.5']} |")
  for kind in ("selected", "final"):
    L += ["", f"## Held-out test, {kind} checkpoints (seeds 5001+5002, 2048 envs per policy, 30 s)", "",
          "| preset | metric | A seeds 1/2/3 | A mean | C seeds 1/2/3 | C mean | C-A | bootstrap 95% CI | Welch 95% CI | perm p (C>A) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for preset, res in out["kinds"][kind].items():
      if res == "incomplete":
        L.append(f"| {preset} | incomplete | | | | | | | | |")
        continue
      for m in ("turns", "turns_with_resets", "mean_yaw_rate", "survival_s", "failure"):
        s = res[m]
        L.append(f"| {preset} | {m} | {' / '.join(f3(x) for x in s['A_seed_means'])} | {f3(s['A_mean'])} | "
                 f"{' / '.join(f3(x) for x in s['C_seed_means'])} | {f3(s['C_mean'])} | {s['diff_C_minus_A']:+.3f} | "
                 f"[{s['boot_ci95'][0]:+.3f}, {s['boot_ci95'][1]:+.3f}] | [{s['welch_ci95'][0]:+.3f}, {s['welch_ci95'][1]:+.3f}] | "
                 f"{s['perm_p_one_sided_C_gt_A']:.2f} |")
    L += ["", f"### Secondary metrics, {kind} checkpoints, nominal (condition means of seed means)", "",
          "| metric | A | C |", "|---|---|---|"]
    res = out["kinds"][kind]["nominal"]
    if res != "incomplete":
      for m in METRICS:
        if m in ("turns", "turns_with_resets", "mean_yaw_rate", "survival_s", "failure"):
          continue
        s = res[m]
        if m == "max_cmd_step_rad":
          L.append(f"| {m} (max over all envs) | {s['A_max']:.4f} | {s['C_max']:.4f} |")
        else:
          L.append(f"| {m} | {s['A_mean']:.4f} | {s['C_mean']:.4f} |")
  L += ["", "## Estimator B audit (closed loop, nominal test rollouts of the selected policies)", "",
        "| source | pos RMSE mm | rot err deg | lin vel RMSE | ang vel RMSE | feature RMSE | contact F1 | mean abs z |", "|---|---|---|---|---|---|---|---|"]
  for k, v in out["estimator_audit"].items():
    if v:
      L.append(f"| {k} | {v['pos_rmse_mm']:.2f} | {v['rot_err_sym_mean_deg']:.2f} | {v['lin_vel_rmse_mps']:.4f} | "
               f"{v['ang_vel_rmse_radps']:.3f} | {v['feature_rmse_all']:.3f} | {v['contact_f1_mean']:.3f} | "
               f"{v.get('input_mean_abs_z', float('nan')):.3f} |")
  (E / "results" / "tables.md").write_text("\n".join(L) + "\n")


def figures(conv, out):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
                       "grid.linewidth": 0.6, "axes.edgecolor": "#8a8a85", "font.size": 9, "figure.facecolor": "#fcfcfb",
                       "axes.facecolor": "#fcfcfb"})
  fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
  for i, c in enumerate("AC"):
    for s, ls in zip(SEEDS, ("-", "--", ":")):
      v = conv["runs"][f"{c}{s}"]
      ax[i].plot(v["iterations"], v["V_smooth"], ls, color=COL[c], lw=2, label=f"{c} seed {s}")
      ax[i].plot(v["iterations"], v["V"], "o", color=COL[c], ms=2.5, alpha=0.35)
      ax[i].plot([v["selected_iteration"]], [v["selected_V"]], "o", ms=8, mfc="none", mec="#3d3d3a", mew=1.2)
    ax[i].axvspan(2000, 3000, color="#efeee9", zorder=0)
    ax[i].set_title(f"Condition {c}: validation turns, 3-pt smoothed (circle = selected)", fontsize=9)
    ax[i].set_xlabel("PPO update")
    ax[i].legend(frameon=False, loc="lower right")
  ax[0].set_ylabel("mean uninterrupted turns in 30 s")
  fig.tight_layout()
  fig.savefig(E / "figures" / "learning_curves.png", dpi=150)
  plt.close(fig)

  res = out["kinds"]["selected"]
  pres = [p for p in PRESETS if res.get(p) not in (None, "incomplete")]
  if not pres:
    return
  fig, ax = plt.subplots(figsize=(11, 3.8))
  for j, p in enumerate(pres):
    t = res[p]["turns"]
    for k, c in enumerate("AC"):
      x = j + (-0.15 if c == "A" else 0.15)
      ms = t[f"{c}_seed_means"]
      ax.plot([x] * 3, ms, "o", color=COL[c], ms=8, mec="#fcfcfb", mew=1.5, label=f"{c} (per training seed)" if j == 0 else None)
      ax.plot([x - 0.08, x + 0.08], [np.mean(ms)] * 2, color="#3d3d3a", lw=2)
    ax.text(j, max(max(t["A_seed_means"]), max(t["C_seed_means"])) + 0.06,
            f"C−A {t['diff_C_minus_A']:+.2f}\n[{t['boot_ci95'][0]:+.2f}, {t['boot_ci95'][1]:+.2f}]", ha="center", fontsize=8, color="#3d3d3a")
  ax.set_xticks(range(len(pres)), pres)
  ax.set_ylabel("mean uninterrupted turns in 30 s")
  ax.set_title("Held-out test, selected checkpoints (seeds 5001+5002); bar = condition mean, text = C−A with bootstrap 95% CI")
  ax.legend(frameon=False, loc="lower left")
  lo = min(min(res[p]["turns"]["A_seed_means"] + res[p]["turns"]["C_seed_means"]) for p in pres)
  hi = max(max(res[p]["turns"]["A_seed_means"] + res[p]["turns"]["C_seed_means"]) for p in pres)
  ax.set_ylim(max(0, lo - 0.15), hi + 0.25)
  fig.tight_layout()
  fig.savefig(E / "figures" / "test_comparison.png", dpi=150)
  plt.close(fig)


if __name__ == "__main__":
  main()
