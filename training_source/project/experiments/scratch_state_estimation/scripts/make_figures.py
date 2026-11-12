#!/usr/bin/env python3
"""Figures and tables for the from-scratch study report.

Reads TensorBoard logs (training), results/curves/*.json (validation learning
curves), results/final/*.json + .npz (held-out test), results/ablation/*.json
and the estimator reports, and writes figures/ and results/tables.md.

    python make_figures.py --exp experiments/scratch_state_estimation
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"A": "#4C72B0", "C": "#DD8452", "D": "#55A868", "P1": "#C44E52", "P2": "#4C72B0", "P3": "#8172B3"}
STEPS_PER_ITER = 4096 * 32
STUDY_LAST_ITER = 1000  # final update of the 1001-iteration budget (model_1000.pt)
# Only the valid study runs; archived dirs (scratch_C_INVALID_*, scratch_C_smoke, *_INTERRUPTED) are excluded.
STUDY_DIRS = ("scratch_A", "scratch_C", "scratch_D")


def tb_scalars(run_dirs: list[str], tags: list[str], max_iter: int | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
  """Concatenate TB logs of one run (later files override earlier steps); keep steps <= max_iter.

  A/D were launched with a 1500-iteration cap and stopped by a watcher after model_1000, so their
  logs may extend slightly past the study budget; truncating at STUDY_LAST_ITER keeps A/C/D comparable.
  """
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  out: dict[str, dict[int, float]] = defaultdict(dict)
  for d in run_dirs:
    for f in sorted(glob.glob(str(Path(d) / "events*"))):
      ea = EventAccumulator(f, size_guidance={"scalars": 0})
      ea.Reload()
      have = ea.Tags()["scalars"]
      for t in tags:
        if t in have:
          for e in ea.Scalars(t):
            if max_iter is None or e.step <= max_iter:
              out[t][e.step] = e.value
  return {t: (np.array(sorted(v)), np.array([v[k] for k in sorted(v)])) for t, v in out.items()}


def smooth(y, k=15):
  if len(y) < k:
    return y
  return np.convolve(y, np.ones(k) / k, mode="valid")


def runs_by_condition(logroot: Path) -> dict[str, dict[int, list[str]]]:
  out: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
  dirs = [d for sub in STUDY_DIRS for d in glob.glob(str(logroot / sub / "*_study_*_seed*"))]
  for d in sorted(dirs):
    m = re.search(r"study_([ACD])_seed(\d+)", d)
    if m:
      out[m.group(1)][int(m.group(2))].append(d)
  return out


def plot_training(runs, figdir: Path) -> dict:
  panels = [
    ("Train/mean_reward", "mean episode reward (training)"),
    ("Episode_Metrics/genuine_yaw_rate", "genuine +Z yaw rate [rad/s] (training episodes)"),
    ("Train/mean_episode_length", "mean episode length [steps]"),
    ("Episode_Termination/cube_pose_deviation", "pose-deviation terminations / step"),
    ("Episode_Termination/cube_fell", "drop terminations / step"),
    ("Loss/exec_saturation", "executed-action saturation (|a|>0.99)"),
    ("Loss/effective_exploration", "effective exploration share"),
    ("Loss/learning_rate", "learning rate"),
    ("Loss/approx_kl", "approx. KL after update"),
    ("Loss/action_std_mean", "policy std"),
    ("Loss/entropy", "entropy"),
    ("Loss/grad_norm_actor", "actor gradient norm"),
  ]
  tags = [p[0] for p in panels]
  fig, axes = plt.subplots(3, 4, figsize=(20, 12))
  summary = {}
  for cond, seeds in sorted(runs.items()):
    for seed, dirs in sorted(seeds.items()):
      sc = tb_scalars(dirs, tags, STUDY_LAST_ITER)
      summary[f"{cond}_seed{seed}"] = {t: float(np.mean(v[1][-20:])) for t, v in sc.items() if len(v[1])}
      for ax, (t, label) in zip(axes.flat, panels):
        if t not in sc:
          continue
        x, y = sc[t]
        ys = smooth(y)
        ax.plot((x[len(x) - len(ys):] + 1) * STEPS_PER_ITER / 1e6, ys, color=COLORS[cond], alpha=0.8,
                label=f"{cond} seed {seed}")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("environment steps [M]", fontsize=8)
        if t == "Loss/learning_rate":
          ax.set_yscale("log")
  axes.flat[0].legend(fontsize=7)
  fig.suptitle("PPO training diagnostics (smoothed; 1 iteration = 131k env steps)")
  fig.tight_layout()
  fig.savefig(figdir / "training_curves.png", dpi=110)
  plt.close(fig)
  return summary


def plot_pilot(logroot: Path, figdir: Path) -> None:
  dirs = {"P1": glob.glob(str(logroot / "scratch_A_clipaction" / "*pilot_P1*")),
          "P2": glob.glob(str(logroot / "scratch_A" / "*pilot_P2*")),
          "P3": glob.glob(str(logroot / "scratch_A" / "*pilot_P3*"))}
  labels = {"P1": "P1 clip + adaptive LR (original)", "P2": "P2 tanh + adaptive LR", "P3": "P3 tanh + fixed 3e-4"}
  panels = [("Train/mean_reward", "reward"), ("Episode_Metrics/genuine_yaw_rate", "genuine yaw rate [rad/s]"),
            ("Loss/exec_saturation", "executed saturation"), ("Loss/exec_mean_saturation", "mean-action saturation"),
            ("Loss/raw_mean_abs_mean", "mean |mu| (pre-transform)"), ("Loss/learning_rate", "learning rate"),
            ("Loss/approx_kl", "approx KL"), ("Loss/effective_exploration", "effective exploration")]
  fig, axes = plt.subplots(2, 4, figsize=(20, 8))
  for k, ds in dirs.items():
    if not ds:
      continue
    sc = tb_scalars(ds, [p[0] for p in panels])
    for ax, (t, lab) in zip(axes.flat, panels):
      if t in sc:
        x, y = sc[t]
        ys = smooth(y, 9)
        ax.plot(x[len(x) - len(ys):], ys, color=COLORS[k], label=labels[k])
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel("iteration", fontsize=8)
  axes.flat[0].legend(fontsize=8)
  fig.suptitle("Stage-0 pilot: action transform and learning-rate schedule (seed 1, 200 iterations)")
  fig.tight_layout()
  fig.savefig(figdir / "pilot_stage0.png", dpi=110)
  plt.close(fig)


def plot_validation_curves(curvedir: Path, figdir: Path) -> dict:
  data: dict[str, dict[int, dict]] = defaultdict(dict)
  sel = {}
  for f in glob.glob(str(curvedir / "*.json")):
    r = json.loads(Path(f).read_text())
    m = re.fullmatch(r"([ACD])_seed(\d+)", Path(f).stem)
    if not m:
      continue
    data[m.group(1)][int(m.group(2))] = r["points"]
    sel[Path(f).stem] = r.get("selected_iteration")
  metrics = [("turns", "uninterrupted turns in 30 s"), ("failure", "failure rate within 30 s"),
             ("survival_s", "survival [s]"), ("mean_yaw_rate", "mean +Z yaw rate [rad/s]")]
  fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
  for cond, seeds in sorted(data.items()):
    for ax, (k, lab) in zip(axes, metrics):
      allits = sorted({int(i) for pts in seeds.values() for i in pts})
      for seed, pts in sorted(seeds.items()):
        its = sorted(int(i) for i in pts)
        ax.plot(np.array(its) * STEPS_PER_ITER / 1e6, [pts[str(i)][k]["mean"] for i in its], color=COLORS[cond], alpha=0.35, lw=1)
      common = [i for i in allits if all(str(i) in p for p in seeds.values())]
      if common:
        m = np.array([[seeds[s][str(i)][k]["mean"] for i in common] for s in seeds])
        ax.plot(np.array(common) * STEPS_PER_ITER / 1e6, m.mean(0), color=COLORS[cond], lw=2.5, label=f"{cond} (mean of {len(seeds)} seeds)")
      ax.set_title(lab, fontsize=10)
      ax.set_xlabel("environment steps [M]")
  axes[0].legend(fontsize=8)
  fig.suptitle("Validation learning curves (seed 7001, nominal, 1024 envs; thin = individual seeds)")
  fig.tight_layout()
  fig.savefig(figdir / "validation_curves.png", dpi=110)
  plt.close(fig)
  return sel


FINAL_METRICS = [("turns", "turns (uninterrupted)"), ("turns_with_resets", "turns incl. after resets"),
                 ("positive_rotation_rad", "positive-direction rotation [rad]"), ("mean_yaw_rate", "mean yaw rate [rad/s]"),
                 ("drop", "drop rate"), ("pose_failure", "tilt/drift failure rate"), ("survival_s", "survival [s]"),
                 ("drift_mean_mm", "mean drift [mm]"), ("tilt_mean_deg", "mean tilt [deg]"), ("reward_sum", "episode reward"),
                 ("exec_saturation", "executed saturation")]


def load_final(finaldir: Path) -> dict:
  """-> res[cond][preset][seed_name] = per-env arrays (both test seeds pooled)."""
  res: dict = defaultdict(lambda: defaultdict(dict))
  for f in glob.glob(str(finaldir / "*.json")):
    r = json.loads(Path(f).read_text())
    name = Path(f).stem
    cond = r["cond"]
    presets = sorted({k.split("/")[0] for k in r["results"]})
    for pr in presets:
      arrs = [np.load(z) for z in sorted(glob.glob(str(finaldir / f"{name}__{pr}__seed*.npz")))]
      if arrs:
        res[cond][pr][name] = {k: np.concatenate([a[k] for a in arrs]) for k in arrs[0].files}
  return res


def final_tables(res: dict, figdir: Path) -> str:
  lines = ["## Held-out test results\n",
           "Mean over training seeds of the per-seed means (each per-seed mean pools both test seeds x all envs). "
           "± is the across-training-seed standard deviation; [lo, hi] is a 95% bootstrap CI over training seeds "
           "(n = number of seeds; wide when n is small).\n"]
  presets = sorted({p for c in res.values() for p in c})
  rng = np.random.default_rng(0)
  table = {}
  for pr in presets:
    lines.append(f"\n### preset `{pr}`\n")
    conds = [c for c in ("A", "C", "D") if pr in res.get(c, {})]
    lines.append("| metric | " + " | ".join(f"{c} (n={len(res[c][pr])})" for c in conds) + " |")
    lines.append("|---|" + "---|" * len(conds))
    for k, lab in FINAL_METRICS:
      row = []
      for c in conds:
        per_seed = np.array([float(np.mean(v[k])) for v in res[c][pr].values()])
        boots = [rng.choice(per_seed, len(per_seed)).mean() for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        sd = per_seed.std(ddof=1) if len(per_seed) > 1 else 0.0
        table.setdefault(pr, {}).setdefault(c, {})[k] = {"mean": float(per_seed.mean()), "sd": float(sd), "ci": [float(lo), float(hi)], "per_seed": per_seed.tolist()}
        row.append(f"{per_seed.mean():.3f} ± {sd:.3f} [{lo:.3f}, {hi:.3f}]")
      lines.append(f"| {lab} | " + " | ".join(row) + " |")
  # bar figure for the main metrics across presets
  keys = ["turns", "drop", "pose_failure", "survival_s"]
  fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4.5))
  for ax, k in zip(axes, keys):
    w = 0.25
    for j, c in enumerate(("A", "C", "D")):
      means, errs, xs = [], [], []
      for i, pr in enumerate(presets):
        if c in table.get(pr, {}):
          t = table[pr][c][k]
          xs.append(i + (j - 1) * w)
          means.append(t["mean"])
          errs.append(t["sd"])
          ax.scatter([i + (j - 1) * w] * len(t["per_seed"]), t["per_seed"], color="k", s=6, zorder=3)
      ax.bar(xs, means, w, yerr=errs, color=COLORS[c], label=c, capsize=3)
    ax.set_xticks(range(len(presets)), presets, rotation=25, fontsize=8)
    ax.set_title(dict(FINAL_METRICS)[k], fontsize=10)
  axes[0].legend()
  fig.suptitle("Held-out test (bars: mean over training seeds, error: seed s.d., dots: individual seeds)")
  fig.tight_layout()
  fig.savefig(figdir / "final_comparison.png", dpi=110)
  plt.close(fig)
  (figdir.parent / "results" / "final_table.json").write_text(json.dumps(table, indent=2))
  return "\n".join(lines) + "\n"


def ablation_table(abldir: Path, figdir: Path) -> str:
  lines = ["## Feature ablation (C and D)\n", "Paired per-env differences vs. the normal run (same seed/preset/randomisation), "
           "mean ± 95% CI over envs; executed-action change measured on visited states.\n"]
  rows = []
  for f in sorted(glob.glob(str(abldir / "*.json"))):
    r = json.loads(Path(f).read_text())
    name = Path(f).stem
    keys = sorted({"/".join(k.split("/")[:2]) for k in r["results"]})
    for ps in keys:
      pr, seed = ps.split("/")
      base_npz = abldir / f"{name}__{pr}__{seed}__none.npz"
      if not base_npz.exists():
        continue
      b = np.load(base_npz)
      sens = r["results"].get(f"{ps}/sensitivity", {})
      for mode in ("zero", "mean", "shuffle"):
        z = abldir / f"{name}__{pr}__{seed}__{mode}.npz"
        if not z.exists():
          continue
        a = np.load(z)
        cells = []
        for k in ("turns", "failure", "drop", "survival_s"):
          d = a[k].astype(float) - b[k].astype(float)
          cells.append(f"{d.mean():+.3f} ± {1.96 * d.std(ddof=1) / math.sqrt(len(d)):.3f}")
        s = sens.get(mode, {})
        rows.append(f"| {name} | {pr} | {seed} | {mode} | " + " | ".join(cells) +
                    f" | {s.get('mean_abs', float('nan')):.4f} | {s.get('frac_gt_0.05', float('nan')):.3f} | {float(np.mean(b['turns'])):.3f} |")
  lines.append("| policy | preset | seed | ablation | Δ turns | Δ failure | Δ drop | Δ survival [s] | mean abs Δ executed action | share Δ>0.05 | normal turns |")
  lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
  lines += rows
  return "\n".join(lines) + "\n"


def pairwise_table(res: dict) -> str:
  """Condition differences with the training seed as the unit (independent seeds; no pairing across conditions).

  Per preset and pooled over presets (per-seed mean over the six presets). 95% CI: bootstrap resampling of the
  per-seed means within each condition independently (2000 draws). n = 3 seeds per condition, so CIs are wide and
  approximate; a CI excluding 0 is reported descriptively, not as a formal significance test.
  """
  rng = np.random.default_rng(1)
  presets = sorted({p for c in res.values() for p in c})
  lines = ["## Pairwise condition differences (training seed = unit)\n",
           "Difference of across-seed means, 95% bootstrap CI over training seeds (independent resampling in each "
           "condition). `pooled` = per-seed mean over all six presets.\n",
           "| metric | preset | C - A | D - A | D - C |", "|---|---|---|---|---|"]
  out = {}
  for k in ("turns", "failure", "survival_s"):
    for pr in presets + ["pooled"]:
      per = {}
      for c in ("A", "C", "D"):
        if pr == "pooled":
          names = sorted(set.intersection(*[set(res[c][q]) for q in presets])) if c in res else []
          per[c] = np.array([np.mean([float(np.mean(res[c][q][n][k])) for q in presets]) for n in names])
        elif c in res and pr in res[c]:
          per[c] = np.array([float(np.mean(v[k])) for v in res[c][pr].values()])
      cells = []
      for x, y in (("C", "A"), ("D", "A"), ("D", "C")):
        if len(per.get(x, [])) and len(per.get(y, [])):
          d = per[x].mean() - per[y].mean()
          bs = [rng.choice(per[x], len(per[x])).mean() - rng.choice(per[y], len(per[y])).mean() for _ in range(2000)]
          lo, hi = np.percentile(bs, [2.5, 97.5])
          out.setdefault(k, {}).setdefault(pr, {})[f"{x}-{y}"] = {"diff": float(d), "ci": [float(lo), float(hi)]}
          cells.append(f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        else:
          cells.append("n/a")
      lines.append(f"| {k} | {pr} | " + " | ".join(cells) + " |")
  return "\n".join(lines) + "\n", out


ECL_KEYS = [("pos_rmse_mm", "pos RMSE [mm]"), ("rot_err_sym_mean_deg", "sym. rot err [deg]"),
            ("lin_vel_rmse_mps", "lin vel RMSE [m/s]"), ("ang_vel_rmse_radps", "ang vel RMSE [rad/s]"),
            ("yaw_rate_rmse_radps", "yaw-rate RMSE [rad/s]")]


def closed_loop_estimator_table(finaldir: Path) -> str:
  """Estimator B scored on the states each policy actually visits (held-out test runs).

  A seeds 2 and 3 generated B's train/val data and A seed 1 its test data, so A rows are in-distribution
  (different random seeds/presets); C and D rows measure the shift induced by a different policy.
  """
  agg: dict = defaultdict(lambda: defaultdict(list))
  for f in sorted(glob.glob(str(finaldir / "*.json"))):
    r = json.loads(Path(f).read_text())
    name = Path(f).stem
    for key, v in r["results"].items():
      e = v.get("estimator_closed_loop")
      if not e:
        continue
      pr = key.split("/")[0]
      for k, _ in ECL_KEYS:
        agg[(name, pr)][k].append(e[k])
  lines = ["## Estimator B accuracy in closed loop (held-out test rollouts of each policy)\n",
           "Per run: mean over the two test seeds. Nominal preset and mean over all six presets. A seeds 2/3 produced "
           "B's training/validation data and A seed 1 its offline test data.\n",
           "| run | scope | " + " | ".join(l for _, l in ECL_KEYS) + " |", "|---|---|" + "---|" * len(ECL_KEYS)]
  names = sorted({n for n, _ in agg})
  for n in names:
    prs = sorted({p for m, p in agg if m == n})
    nom = [f"{np.mean(agg[(n, 'nominal')][k]):.3f}" if (n, "nominal") in agg else "n/a" for k, _ in ECL_KEYS]
    allp = [f"{np.mean([np.mean(agg[(n, p)][k]) for p in prs]):.3f}" for k, _ in ECL_KEYS]
    lines.append(f"| {n} | nominal | " + " | ".join(nom) + " |")
    lines.append(f"| {n} | all presets | " + " | ".join(allp) + " |")
  return "\n".join(lines) + "\n"


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--exp", default="experiments/scratch_state_estimation")
  p.add_argument("--logroot", default="logs/rsl_rl")
  a = p.parse_args()
  exp = Path(a.exp)
  figdir = exp / "figures"
  figdir.mkdir(parents=True, exist_ok=True)
  (exp / "results").mkdir(exist_ok=True)
  logroot = Path(a.logroot)
  md = ["# Generated tables\n"]
  plot_pilot(logroot, figdir)
  runs = runs_by_condition(logroot)
  if runs:
    s = plot_training(runs, figdir)
    (exp / "results" / "training_last20_summary.json").write_text(json.dumps(s, indent=2))
  if (exp / "results" / "curves").exists():
    sel = plot_validation_curves(exp / "results" / "curves", figdir)
    md.append("## Selected checkpoints (validation)\n\n" + "\n".join(f"- {k}: iteration {v}" for k, v in sorted(sel.items())) + "\n")
  if (exp / "results" / "final").exists():
    res = load_final(exp / "results" / "final")
    md.append(final_tables(res, figdir))
    pw_md, pw = pairwise_table(res)
    md.append(pw_md)
    (exp / "results" / "pairwise_differences.json").write_text(json.dumps(pw, indent=2))
    md.append(closed_loop_estimator_table(exp / "results" / "final"))
  if (exp / "results" / "ablation").exists():
    md.append(ablation_table(exp / "results" / "ablation", figdir))
  (exp / "results" / "tables.md").write_text("\n".join(md))
  print("\n".join(md))


if __name__ == "__main__":
  main()
