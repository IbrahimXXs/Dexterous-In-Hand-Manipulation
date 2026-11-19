#!/usr/bin/env python3
"""Phase 4: Retention-V1 vs the hardware-limited A baseline (PROTOCOL.md Sections 3-4).

Inputs: results/test/R*_*.npz (test30.py), ../hwlimit_ac/results/test/A_seed*_*.npz,
data/test60_*.npz (test60.py), results/curves/R1.json + ../hwlimit_ac/results/curves/A*.json,
results/convergence.json, results/diagnosis.json.
Outputs: results/comparison.json, results/comparison.md, figures/cmp_*.png.

Statistics (one Retention-V1 training seed: exploratory):
  * A: per-seed means (3 seeds) and their mean; R1: pooled over test seeds 5001+5002.
  * R1 - A_mean with a bootstrap 95% CI over R1's environments and A's 3 seed means
    (two-level for A), and R1's position within the A seed range.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

E = Path("experiments/retention_v1")
H = Path("experiments/hwlimit_ac")
PRESETS = ("nominal", "heavy_slippery", "light_grippy", "high_delay", "weak_motors", "obs_noise_x3")
PRESETS60 = ("nominal", "high_delay", "heavy_slippery")
A_SEL = {1: "A_seed1_selected_model_2600", 2: "A_seed2_selected_model_2900", 3: "A_seed3_selected_model_2600"}
M30 = ["turns", "failure", "survival_s", "mean_yaw_rate", "turns_with_resets", "completed", "rotating_fraction", "drift_mean_mm",
       "tilt_mean_deg", "tilt_max_deg", "cmd_at_limit_frac", "any_joint_at_limit_frac", "kit_clip_frac", "action_change", "exec_saturation"]
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8a85"
DT = 0.05


def load30(prefix, root, preset):
  arrs = [dict(np.load(root / "results" / "test" / f"{prefix}__{preset}__seed{s}.npz")) for s in (5001, 5002)]
  return {k: np.concatenate([a[k] for a in arrs]) for k in arrs[0]}


def boot_diff(r, a_list, rng, n=4000):
  out = np.empty(n)
  for b in range(n):
    rr = r[rng.integers(0, len(r), len(r))].mean()
    idx = rng.integers(0, 3, 3)
    aa = np.mean([a_list[i][rng.integers(0, len(a_list[i]), len(a_list[i]))].mean() for i in idx])
    out[b] = rr - aa
  return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def long_metrics(tr, mu, ci):
  T, n = tr["drift"].shape
  end = tr["end_step"]
  failed = end >= 0
  turns60 = tr["yaw_inc"].sum(0) / (2 * math.pi)
  pos = tr["cube_pos_palm"].astype(np.float32)
  alive = tr["alive"].astype(bool)
  md = np.sqrt(np.einsum("tni,ij,tnj->tn", pos - mu, ci, pos - mu))
  out_ws = alive & (md >= 3.0)
  first_exit = np.where(out_ws.any(0), out_ws.argmax(0), -1)
  ncont = (tr["distal_force"].astype(np.float32) > 0.05).sum(-1)
  surv = ~failed
  last = np.where(failed, np.maximum(end - 1, 0), T - 1)
  disp = np.linalg.norm(pos[last, np.arange(n)] - pos[0], axis=-1)
  return {
    "turns_30s": float(tr["yaw_inc"][:600].sum(0).mean() / (2 * math.pi)), "turns_60s": float(turns60.mean()),
    "retain_30s": float((~failed | (end >= 600)).mean()), "retain_60s": float(surv.mean()),
    "mean_survival_60s_s": float(np.where(failed, end, T).mean() * DT),
    "failure_time_median_s": float(np.median(end[failed]) * DT) if failed.any() else None,
    "revolutions_before_failure_mean": float(turns60[failed].mean()) if failed.any() else None,
    "yaw_rate_alive_rad_s": float(tr["yaw_inc"].sum() / (alive.sum() * DT)),
    "palm_md_mean_alive": float(md[alive].mean()),
    "share_alive_steps_outside_md3": float(out_ws.sum() / alive.sum()),
    "share_episodes_ever_outside_md3": float((first_exit >= 0).mean()),
    "time_to_leave_md3_median_s": float(np.median(first_exit[first_exit >= 0]) * DT) if (first_exit >= 0).any() else None,
    "palm_displacement_at_end_mm_median": float(np.median(disp) * 1000),
    "distal_contacts_mean_alive": float(ncont[alive].mean()), "share_alive_steps_le1_contact": float((ncont[alive] <= 1).mean()),
    "cmd_at_limit_frac_alive": float(tr["at_limit"][alive].mean()),
    "survival_curve": [float((~failed | (end >= t)).mean()) for t in range(0, T + 1, 20)],
    "cache_id": tr["cache_id"], "s30": (~failed | (end >= 600)),
  }


def main():
  rng = np.random.default_rng(0)
  conv = json.loads((E / "results" / "convergence.json").read_text())["runs"]
  R = "R1"
  out = {"run": R, "convergence": {k: conv[R].get(k) for k in ("status", "selected_iteration", "selected_V", "final_V",
                                                               "learning_speed_mean_V_100_3000", "first_iteration_V_ge_0.5", "criteria")},
         "test30": {}, "test60": {}}
  for kind in ("selected", "final"):
    out["test30"][kind] = {}
    it = conv[R]["selected_iteration"] if kind == "selected" else 3000
    for preset in PRESETS:
      r = load30(f"{R}_{kind}_model_{it}", E, preset)
      A = [load30(A_SEL[s] if kind == "selected" else f"A_seed{s}_final_model_3000", H, preset) for s in (1, 2, 3)]
      res = {}
      for m in M30:
        rv = r[m].astype(np.float64)
        av = [a[m].astype(np.float64) for a in A]
        am = [float(x.mean()) for x in av]
        d = {"R1": float(rv.mean()), "A_seeds": am, "A_mean": float(np.mean(am)), "diff": float(rv.mean() - np.mean(am)),
             "rel_change": float(rv.mean() / np.mean(am) - 1) if np.mean(am) else None}
        if m in ("turns", "failure", "survival_s", "mean_yaw_rate", "turns_with_resets"):
          d["boot_ci95"] = boot_diff(rv, av, rng)
          d["R1_rank_among_A_seeds"] = int(sum(rv.mean() > x for x in am))  # 0..3
        res[m] = d
      out["test30"][kind][preset] = res
  # targets (selected, nominal)
  n = out["test30"]["selected"]["nominal"]
  diag = json.loads((E / "results" / "diagnosis.json").read_text())["pooled"]
  mu = np.array(diag["palm_frame_survivor_mean_full_m"]); ci = np.linalg.inv(np.array(diag["palm_frame_survivor_cov_m2"]))
  for preset in PRESETS60:
    res = {}
    for tag in ("A1", "A2", "A3", f"{R}_selected"):
      f = E / "data" / f"test60_{tag}_{preset}_seed5001.npz"
      if f.exists():
        d = dict(np.load(f))
        res[tag] = long_metrics({k: (v.astype(np.float32) if v.dtype == np.float16 else v) for k, v in d.items()}, mu, ci)
    out["test60"][preset] = res
  t60 = out["test60"]["nominal"]
  a60 = [t60[a] for a in ("A1", "A2", "A3") if a in t60]
  r60 = t60.get(f"{R}_selected")
  targets = {
    "T1_turns30_ge_1.39": n["turns"]["R1"] >= 1.39,
    "T2_failure_lt_0.193": n["failure"]["R1"] < 0.193,
    "T3_survival_gt_24.7": n["survival_s"]["R1"] > 24.7,
    "T4_turns60_and_retain60_above_A_mean": bool(r60 and r60["turns_60s"] > np.mean([a["turns_60s"] for a in a60])
                                                 and r60["retain_60s"] > np.mean([a["retain_60s"] for a in a60])),
    "T5_yaw_rate_ge_0.9_A": n["mean_yaw_rate"]["R1"] >= 0.9 * n["mean_yaw_rate"]["A_mean"],
  }
  out["targets"] = {k: bool(v) for k, v in targets.items()}
  passed = sum(targets.values())
  if passed == 5:
    outcome = "A (promising)"
  elif not (targets["T1_turns30_ge_1.39"] and targets["T5_yaw_rate_ge_0.9_A"]) and (targets["T2_failure_lt_0.193"] or targets["T3_survival_gt_24.7"]):
    outcome = "B (retention up, rotation down)" if not targets["T5_yaw_rate_ge_0.9_A"] or n["turns"]["diff"] < 0 else "C (no improvement)"
  else:
    outcome = "C (no improvement)"
  out["outcome_rule"] = outcome
  # per-grasp (60-s nominal traces): does R1 fix the grasps that are bad for A?
  if r60 and a60:
    cidA = np.concatenate([a["cache_id"] for a in a60]); sA = np.concatenate([a["s30"] for a in a60])
    cidR, sR = r60["cache_id"], r60["s30"]
    g = np.intersect1d(np.unique(cidA), np.unique(cidR))
    pa = np.array([sA[cidA == x].mean() for x in g]); pr = np.array([sR[cidR == x].mean() for x in g])
    bad = pa < 0.5
    out["per_grasp"] = {"grasps": int(len(g)), "corr_A_vs_R1": float(np.corrcoef(pa, pr)[0, 1]),
                        "A_bad_grasps_n": int(bad.sum()), "A_bad_grasps_A_survival": float(pa[bad].mean()), "A_bad_grasps_R1_survival": float(pr[bad].mean()),
                        "A_good_grasps_A_survival": float(pa[~bad].mean()), "A_good_grasps_R1_survival": float(pr[~bad].mean())}
  for p_ in out["test60"].values():
    for v in p_.values():
      v.pop("cache_id", None); v.pop("s30", None)
  (E / "results" / "comparison.json").write_text(json.dumps(out, indent=1))
  write_md(out)
  figures(out, conv)
  print(json.dumps({k: out[k] for k in ("convergence", "targets", "outcome_rule")}, indent=1, default=str))


def write_md(out):
  L = ["# Retention-V1 seed 1 vs hardware-limited A (generated by scripts/compare.py)", ""]
  for kind in ("selected", "final"):
    L += [f"## 30-s test, {kind} checkpoints (seeds 5001+5002, 2048 environments per policy)", "",
          "| preset | metric | A seeds 1/2/3 | A mean | R1 | R1 - A mean [95% CI] | relative | R1 above n of 3 A seeds |", "|---|---|---|---|---|---|---|---|"]
    for preset, res in out["test30"][kind].items():
      for m in ("turns", "failure", "survival_s", "mean_yaw_rate", "turns_with_resets"):
        d = res[m]
        L.append(f"| {preset} | {m} | {' / '.join(f'{x:.3f}' for x in d['A_seeds'])} | {d['A_mean']:.3f} | {d['R1']:.3f} | "
                 f"{d['diff']:+.3f} [{d['boot_ci95'][0]:+.3f}, {d['boot_ci95'][1]:+.3f}] | {100 * d['rel_change']:+.1f}% | {d['R1_rank_among_A_seeds']} |")
    L += ["", f"### Secondary (nominal, {kind})", "", "| metric | A mean | R1 |", "|---|---|---|"]
    for m, d in out["test30"][kind]["nominal"].items():
      if "boot_ci95" not in d:
        L.append(f"| {m} | {d['A_mean']:.4f} | {d['R1']:.4f} |")
  L += ["", "## 60-s retention test (seed 5001, 1024 environments, first episode only)", ""]
  keys = ["turns_30s", "turns_60s", "retain_30s", "retain_60s", "mean_survival_60s_s", "failure_time_median_s", "revolutions_before_failure_mean",
          "yaw_rate_alive_rad_s", "palm_md_mean_alive", "share_alive_steps_outside_md3", "share_episodes_ever_outside_md3", "time_to_leave_md3_median_s",
          "palm_displacement_at_end_mm_median", "distal_contacts_mean_alive", "share_alive_steps_le1_contact", "cmd_at_limit_frac_alive"]
  for preset, res in out["test60"].items():
    tags = list(res)
    L += [f"### {preset}", "", "| metric | " + " | ".join(tags) + " |", "|---" * (len(tags) + 1) + "|"]
    for k in keys:
      L.append(f"| {k} | " + " | ".join("-" if res[t].get(k) is None else f"{res[t][k]:.3f}" for t in tags) + " |")
    L.append("")
  if "per_grasp" in out:
    L += ["## Per initial grasp (60-s nominal, 30-s retention)", "", json.dumps(out["per_grasp"]), ""]
  L += ["## Pre-registered targets (selected, nominal)", "", json.dumps(out["targets"]), "", f"Outcome by the pre-registered rule: **{out['outcome_rule']}**"]
  (E / "results" / "comparison.md").write_text("\n".join(L) + "\n")


def figures(out, conv):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
                       "font.size": 9, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
  # learning curves
  fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
  for i, (key, lab) in enumerate((("turns", "validation turns (30 s)"), ("failure", "validation failure rate"))):
    for s, ls in zip((1, 2, 3), ("-", "--", ":")):
      a = json.loads((H / "results" / "curves" / f"A{s}.json").read_text())["points"]
      its = sorted(map(int, a))
      ax[i].plot(its, [a[str(t)][key]["mean"] for t in its], ls, color=BLUE, lw=1.6, label=f"hwlimit A seed {s}")
    r = json.loads((E / "results" / "curves" / "R1.json").read_text())["points"]
    its = sorted(map(int, r))
    ax[i].plot(its, [r[str(t)][key]["mean"] for t in its], "-", color=ORANGE, lw=2.4, label="Retention-V1 seed 1")
    ax[i].set_xlabel("PPO update"); ax[i].set_title(lab)
  ax[0].legend(frameon=False, loc="lower right")
  fig.tight_layout(); fig.savefig(E / "figures" / "cmp_learning_curves.png", dpi=140); plt.close(fig)
  # 30-s test per preset
  res = out["test30"]["selected"]
  fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
  for j, (m, lab) in enumerate((("turns", "uninterrupted turns (30 s)"), ("failure", "pose-failure rate"), ("survival_s", "survival (s of 30)"))):
    for i, p in enumerate(PRESETS):
      ax[j].plot([i - 0.12] * 3, res[p][m]["A_seeds"], "o", color=BLUE, ms=6, label="A (per seed)" if i == 0 else None)
      ax[j].plot([i + 0.12], [res[p][m]["R1"]], "o", color=ORANGE, ms=8, label="Retention-V1 seed 1" if i == 0 else None)
    ax[j].set_xticks(range(len(PRESETS)), [p.replace("_", "\n") for p in PRESETS], fontsize=7.5); ax[j].set_title(lab)
  ax[0].legend(frameon=False)
  fig.suptitle("Held-out 30-s test, selected checkpoints (seeds 5001+5002)")
  fig.tight_layout(); fig.savefig(E / "figures" / "cmp_test30.png", dpi=140); plt.close(fig)
  # 60-s survival curves
  fig, ax = plt.subplots(1, len(out["test60"]), figsize=(13, 3.6), sharey=True)
  for k, (preset, r60) in enumerate(out["test60"].items()):
    for tag, v in r60.items():
      tt = np.arange(len(v["survival_curve"])) * 1.0
      ax[k].plot(tt, v["survival_curve"], color=ORANGE if tag.startswith("R") else BLUE, lw=2.4 if tag.startswith("R") else 1.3,
                 ls="-" if tag.startswith("R") or tag == "A1" else ("--" if tag == "A2" else ":"), label=tag.replace("_selected", ""))
    ax[k].set_title(f"{preset}: P(cube retained)"); ax[k].set_xlabel("time (s)")
  ax[0].legend(frameon=False); ax[0].set_ylabel("share of episodes")
  fig.tight_layout(); fig.savefig(E / "figures" / "cmp_survival60.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
  main()
