#!/usr/bin/env python3
"""Seed jury, Step 1 figure (light + dark), README style (scripts/make_readme_figures.py THEMES/style unchanged).

Left: smoothed disagreement D̄ over time for the four runs fixed in PLAN.md 1.6, with the drop, the pre-drop window
and the control span marked.  Right: per primary-set run, every control-window score and the pre-drop score,
each divided by that run's control median (1 = no change).

    uv run --no-project --with matplotlib --with numpy \
        python training_source/project/experiments/seed_jury/scripts/figure_real.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

E = Path(__file__).resolve().parents[1]
ROOT = E.parents[3]


def _load(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[name] = mod
  spec.loader.exec_module(mod)
  return mod


MRF = _load("make_readme_figures", ROOT / "scripts" / "make_readme_figures.py")
RR = _load("seed_jury_real_replay", E / "scripts" / "real_replay.py")
plt = MRF.plt


def main() -> None:
  res = json.loads((E / "results" / "real_replay.json").read_text())
  seeds = RR.load_seeds()
  lower, upper = RR.DD._HW_LIMIT_LO + RR.margin(), RR.DD._HW_LIMIT_HI - RR.margin()
  runs = {(r["session"], r["trial"]): r for r in RR.load_runs()}
  P = [r for r in runs.values() if r["dropped"] and r["t_drop"] >= 10]
  need = set(RR.FIG_RUNS) | {(r["session"], r["trial"]) for r in P}
  for k in need:
    runs[k]["rep"] = RR.replay(runs[k], seeds, lower, upper)
  h1, h2 = res["primary"]["H1"], res["primary"]["H2"]
  for name, th in MRF.THEMES.items():
    MRF.style(th)
    fig = plt.figure(figsize=(13.5, 6.0))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.35, 1], hspace=0.55, wspace=0.18, left=0.06, right=0.98, top=0.87, bottom=0.17)
    ymax = 0.26
    for i, key in enumerate(RR.FIG_RUNS):
      r = runs[key]
      ax = fig.add_subplot(gs[i, 0])
      t, Db = r["rep"]["t"], r["rep"]["Dbar"]
      if r["dropped"]:
        td = r["t_drop"]
        ax.axvspan(3, td - 5, color=th["grid"], lw=0, zorder=0)
        ax.axvspan(td - 3, td - 1, color=th["C5"], alpha=0.22, lw=0, zorder=0)
        m = t < td
        ax.plot(t[m], Db[m], color=th["C5"], lw=1.4, zorder=2)
        ax.plot(t[~m], Db[~m], color=th["muted"], lw=1.0, zorder=1)
        ax.axvline(td, color=th["critical"], lw=1.4, zorder=3)
        ax.text(td + 0.6, ymax * 1.0, f"drop {td:g} s", color=th["ink"], fontsize=8.5, va="top")
        ttl = f"{key[0]} trial {key[1]} ({r['seed']}): dropped at {td:g} s"
      else:
        ax.axvspan(3, float(t[-1]), color=th["grid"], lw=0, zorder=0)
        ax.plot(t, Db, color=th["C5"], lw=1.4, zorder=2)
        ttl = f"{key[0]} trial {key[1]} ({r['seed']}): held 60 s"
      ax.set_title(ttl, fontsize=9.5, pad=4)
      ax.set_xlim(0, 61)
      ax.set_ylim(0, ymax)
      ax.set_yticks([0, 0.1, 0.2])
      ax.tick_params(labelsize=8.5)
      if i < 3:
        ax.set_xticklabels([])
    ax.set_xlabel("seconds from policy start", fontsize=9)
    fig.text(0.012, 0.53, "D (rad): executing seed vs median of the three seeds, 0.5-s mean", rotation=90, va="center", fontsize=9, color=th["ink2"])
    fig.text(0.06, 0.955, "Disagreement of the three C5 seeds on the replayed real runs", fontsize=11, fontweight="bold", color=th["ink"])

    ax = fig.add_subplot(gs[:, 1])
    order = sorted(P, key=lambda r: (r["seed"], r["session"], r["trial"]))
    for x, r in enumerate(order):
      td = r["t_drop"]
      ctl = np.array([RR.score(r["rep"], a, b) for a, b in RR.control_windows(r, td - 5)])
      med = float(np.median(ctl))
      jit = (np.arange(len(ctl)) % 5 - 2) * 0.06
      ax.scatter(x + jit, ctl / med, s=14, color=th["muted"], alpha=0.7, lw=0, zorder=2)
      pre = RR.score(r["rep"], td - 3, td - 1) / med
      ax.scatter([x], [pre], s=70, marker="D", color=th["C5"], edgecolor=th["surface"], linewidth=1.5, zorder=3)
    ax.axhline(1.0, color=th["axis"], lw=1, zorder=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{r['session']}\nt{r['trial']} {r['seed']}" for r in order], fontsize=7.8)
    ax.set_ylabel("window mean D / run's control median", fontsize=9)
    ax.set_ylim(0.55, 1.55)
    ax.set_title("Pre-drop window (◆) vs control windows (•), drops ≥ 10 s", fontsize=10)
    ax.text(0.02, 0.97, f"H1: median ratio {h1['median_ratio']:.2f} [{h1['ci95'][0]:.2f}, {h1['ci95'][1]:.2f}], not supported\n"
                        f"H2: AUC {h2['auc']:.2f} [{h2['ci95'][0]:.2f}, {h2['ci95'][1]:.2f}], not supported",
            transform=ax.transAxes, va="top", fontsize=8.5, color=th["ink"])
    fig.text(0.06, 0.045, "Left: grey band = control windows (3 s to drop − 5 s); green band = pre-drop window (drop − 3 s to drop − 1 s); red line = operator drop; grey trace = after the drop (never used).",
             fontsize=7.8, color=th["ink2"])
    fig.text(0.06, 0.018, "Pre-registered (PLAN.md), already-collected logs, n = 8 drops at ≥ 10 s. Replay with the kit's own code; "
             f"gate passed (max raw-action error {res['gate']['raw_max_err']:.1e}).", fontsize=7.5, color=th["muted"])
    suffix = "" if name == "light" else "_dark"
    (E / "figures").mkdir(exist_ok=True)
    fig.savefig(E / "figures" / f"real_replay{suffix}.png", dpi=150)
    plt.close(fig)
  print("wrote", E / "figures")


if __name__ == "__main__":
  main()
