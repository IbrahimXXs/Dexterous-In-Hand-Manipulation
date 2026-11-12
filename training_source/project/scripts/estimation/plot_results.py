#!/usr/bin/env python3
"""Comparison plots and a Markdown table for the estimator study.

Inputs
  --eval    one or more ``results.json`` from evaluate_policies.py (merged)
  --runs    NAME=LOGDIR pairs of RSL-RL TensorBoard run directories, for
            training-curve / sample-efficiency plots

Outputs (in --output-dir)
  summary.md                    table: mean ± 95% CI per policy and preset
  bars_<preset>.png             key metrics per policy
  robustness_<metric>.png       metric vs. randomised cube mass / friction /
                                delay (nominal preset, all seeds pooled)
  training_curves.png           reward, episode length, failure terminations
                                vs. environment steps
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

METRICS = [
  ("turns", "turns in 30 s"),
  ("mean_yaw_rate", "mean +Z yaw rate [rad/s]"),
  ("drop", "cube-drop rate"),
  ("failure", "failure rate (any termination)"),
  ("survival_s", "survival [s]"),
  ("reward_sum", "episode reward"),
  ("drift_max_mm", "max drift [mm]"),
  ("tilt_max_deg", "max tilt [deg]"),
  ("rotating_fraction", "fraction of 1 s windows rotating"),
]


def load_eval(paths: list[str]) -> tuple[dict, dict]:
  """-> per_env[policy][preset] = dict of concatenated arrays, configs."""
  per_env: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
  cfgs = {}
  for p in paths:
    res = json.loads(Path(p).read_text())
    d = Path(p).parent
    cfgs[p] = res["config"]
    for pol, entries in res["results"].items():
      for key in entries:
        if "/" not in key:
          continue
        preset, seed = key.split("/")
        npz = d / f"{pol}__{preset}__{seed.replace('seed_', 'seed')}.npz"
        if not npz.is_file():
          continue
        data = np.load(npz)
        for k in data.files:
          per_env[pol][preset][k].append(data[k])
  out = {
    pol: {pr: {k: np.concatenate(v) for k, v in d.items()} for pr, d in presets.items()}
    for pol, presets in per_env.items()
  }
  return out, cfgs


def mean_ci(x: np.ndarray) -> tuple[float, float]:
  x = x.astype(np.float64)
  return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def write_table(data: dict, out: Path) -> None:
  presets = sorted({pr for d in data.values() for pr in d})
  lines = ["# Policy comparison (mean ± 95% CI over environments)\n"]
  for pr in presets:
    lines.append(f"\n## preset: {pr}\n")
    pols = [p for p in data if pr in data[p]]
    lines.append("| metric | " + " | ".join(pols) + " |")
    lines.append("|---|" + "---|" * len(pols))
    for k, label in METRICS:
      row = []
      for p in pols:
        m, c = mean_ci(data[p][pr][k])
        row.append(f"{m:.3f} ± {c:.3f}")
      lines.append(f"| {label} | " + " | ".join(row) + " |")
    lines.append("| N envs | " + " | ".join(str(len(data[p][pr]["turns"])) for p in pols) + " |")
  (out / "summary.md").write_text("\n".join(lines) + "\n")


def bar_plots(data: dict, out: Path) -> None:
  presets = sorted({pr for d in data.values() for pr in d})
  for pr in presets:
    pols = [p for p in data if pr in data[p]]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, (k, label) in zip(axes.flat, METRICS[:8]):
      vals = [mean_ci(data[p][pr][k]) for p in pols]
      ax.bar(range(len(pols)), [v[0] for v in vals], yerr=[v[1] for v in vals], color=[f"C{i}" for i in range(len(pols))], capsize=4)
      ax.set_xticks(range(len(pols)), pols, rotation=20, fontsize=8)
      ax.set_title(label, fontsize=9)
    fig.suptitle(f"Policy comparison - preset '{pr}' (error bars: 95% CI)")
    fig.tight_layout()
    fig.savefig(out / f"bars_{pr}.png", dpi=110)
    plt.close(fig)


def robustness_plots(data: dict, out: Path, preset: str = "nominal") -> None:
  factors = [("dr_cube_mass", "cube mass [kg]"), ("dr_friction", "friction"), ("dr_action_delay_steps", "actuator delay [physics steps]"), ("dr_motor_damping", "mean joint damping")]
  for k, label in (("failure", "failure rate"), ("turns", "turns in 30 s")):
    fig, axes = plt.subplots(1, len(factors), figsize=(5 * len(factors), 4))
    for ax, (f, flabel) in zip(axes, factors):
      for i, pol in enumerate(data):
        if preset not in data[pol] or f not in data[pol][preset]:
          continue
        x, y = data[pol][preset][f], data[pol][preset][k].astype(float)
        edges = np.quantile(x, np.linspace(0, 1, 6))
        centres, means, cis = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
          m = (x >= lo) & (x <= hi)
          if m.sum() < 20:
            continue
          mu, ci = mean_ci(y[m])
          centres.append(0.5 * (lo + hi))
          means.append(mu)
          cis.append(ci)
        ax.errorbar(centres, means, yerr=cis, marker="o", capsize=3, label=pol, color=f"C{i}")
      ax.set_xlabel(flabel)
      ax.set_ylabel(label)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{label} vs. randomised parameter (preset '{preset}', quintile bins)")
    fig.tight_layout()
    fig.savefig(out / f"robustness_{k}.png", dpi=110)
    plt.close(fig)


def training_curves(runs: dict[str, str], out: Path) -> dict:
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  tags = [
    ("Train/mean_reward", "mean episode reward"),
    ("Train/mean_episode_length", "mean episode length [steps]"),
    ("Episode_Termination/cube_pose_deviation", "pose-deviation terminations / iter"),
    ("Episode_Metrics/rotation_progress", "rotation progress metric"),
  ]
  fig, axes = plt.subplots(1, len(tags), figsize=(5 * len(tags), 4))
  summary = {}
  for i, (name, d) in enumerate(runs.items()):
    files = sorted(glob.glob(str(Path(d) / "events*")))
    if not files:
      continue
    ea = EventAccumulator(files[-1], size_guidance={"scalars": 0})
    ea.Reload()
    fps = {e.step: e.value for e in ea.Scalars("Perf/total_fps")} if "Perf/total_fps" in ea.Tags()["scalars"] else {}
    summary[name] = {}
    for ax, (tag, label) in zip(axes, tags):
      if tag not in ea.Tags()["scalars"]:
        continue
      ev = ea.Scalars(tag)
      steps = np.array([e.step for e in ev])
      vals = np.array([e.value for e in ev])
      k = max(1, len(vals) // 30)
      smooth = np.convolve(vals, np.ones(k) / k, mode="valid")
      ax.plot(steps[k - 1 :] - steps[0], smooth, label=name, color=f"C{i}")
      ax.set_xlabel("PPO iterations since fine-tuning start")
      ax.set_title(label, fontsize=9)
      summary[name][tag] = {"first10_mean": float(vals[:10].mean()), "last10_mean": float(vals[-10:].mean()), "iterations": int(len(vals))}
    summary[name]["mean_fps"] = float(np.mean(list(fps.values()))) if fps else None
  axes[0].legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(out / "training_curves.png", dpi=110)
  plt.close(fig)
  (out / "training_summary.json").write_text(json.dumps(summary, indent=2))
  return summary


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--eval", nargs="*", default=[])
  p.add_argument("--runs", nargs="*", default=[], help="NAME=TB_RUN_DIR")
  p.add_argument("--output-dir", required=True)
  a = p.parse_args()
  out = Path(a.output_dir)
  out.mkdir(parents=True, exist_ok=True)
  if a.eval:
    data, _ = load_eval(a.eval)
    write_table(data, out)
    bar_plots(data, out)
    robustness_plots(data, out)
    print((out / "summary.md").read_text())
  if a.runs:
    s = training_curves(dict(r.split("=", 1) for r in a.runs), out)
    print(json.dumps(s, indent=2))
  print(f"[plots] {out}")


if __name__ == "__main__":
  main()
