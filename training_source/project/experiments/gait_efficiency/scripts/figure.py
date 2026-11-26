#!/usr/bin/env python3
"""Gait efficiency figure (light + dark), in the README figure style (scripts/make_readme_figures.py: THEMES, FONT,
style(), _spread() reused unchanged).

Panel 1: finger rhythm (FFT peak, Hz), simulation -> real hand, one line per policy.
Panel 2: quarter turns per finger cycle, simulation -> real hand.
Panel 3: 25 Nov trial 1 (C5 seed 3): cumulative finger cycles (joint encoders) and camera quarter turns over time.
Simulation = setting K (kit starts, 350 mA-equivalent actuator), environments held >= 20 s (pre-registered);
real = runs held >= 20 s, window 3 s to the drop (post hoc).  R1 has no such run and is not drawn.

    uv run --no-project --with matplotlib --with numpy --with scipy python \
        training_source/project/experiments/gait_efficiency/scripts/figure.py
Writes figures/gait_efficiency.png and figures/gait_efficiency_dark.png.
"""

from __future__ import annotations

import csv
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
RH = _load("gait_real_hand", E / "scripts" / "real_hand.py")
plt = MRF.plt

# policy -> (theme colour key, marker, line style); identity is never colour alone (marker + direct label)
POL = {"C5s3": ("C5", "^", "-"), "C5s1": ("C5", "o", "-"), "M400": ("REF", "D", (0, (3, 2))),
       "A3": ("A", "o", "-"), "P0": ("RET", "s", "-")}


def trial1_series() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  runs = RH.load_runs()
  r = next(x for x in runs if x["session"] == "25 Nov" and x["trial"] == 1)
  c = RH.G.count_cycles(r["q"], r["t"], t_end_s=r["held_s"])
  cam = list(csv.DictReader(open(RH.LOGS / "trials_20261125_trial1_vision.csv", newline="")))
  tc = np.array([float(x["t_from_policy_start_s"]) for x in cam])
  qc = np.array([float(x["cumulative_quarter_turns"]) for x in cam])
  qc = qc * (1.0 if qc[-1] >= 0 else -1.0)
  return np.array(c["t_series"]), np.array(c["cum_cycles_series"]), tc, qc


def slope_panel(ax, th, vals_sim, vals_real, fmt, ylim, title, ylab):
  pols = [p for p in POL if p in vals_real]
  gap = (ylim[1] - ylim[0]) * 0.06
  lpos = MRF._spread([vals_sim[p] for p in pols], gap)
  rpos = MRF._spread([vals_real[p] for p in pols], gap)
  for p, ly, ry in zip(pols, lpos, rpos):
    fam, mk, ls = POL[p]
    ax.plot([0, 1], [vals_sim[p], vals_real[p]], color=th[fam], lw=2, ls=ls, zorder=2)
    ax.scatter([0, 1], [vals_sim[p], vals_real[p]], s=46, marker=mk, color=th[fam], zorder=3,
               edgecolor=th["surface"], linewidth=1.5)
    ax.text(-0.08, ly, f"{fmt(vals_sim[p])}  {p}", ha="right", va="center", fontsize=9, color=th["ink"])
    ax.text(1.08, ry, f"{p}  {fmt(vals_real[p])}", va="center", fontsize=9, color=th["ink"])
  ax.set_xlim(-0.75, 1.75)
  ax.set_ylim(*ylim)
  ax.set_xticks([0, 1])
  ax.set_xticklabels(["Simulation", "Real hand"])
  ax.tick_params(axis="x", length=0, labelsize=10)
  ax.set_yticks([])
  for sp in ("left", "bottom"):
    ax.spines[sp].set_visible(False)
  ax.set_title(title)
  ax.text(0.5, -0.16, ylab, transform=ax.transAxes, ha="center", fontsize=9, color=th["ink2"])


def main() -> None:
  dec = json.loads((E / "results" / "decomposition.json").read_text())["policies"]
  real = json.loads((E / "results" / "real_hand.json").read_text())
  cam1 = real["camera_check_25nov"]["trial1"]
  P = {p: r for p, r in dec.items() if p in POL and "r_tpc" in r}
  rh_s = {p: r["sim_K_effort"]["rhythm_hz_median"] for p, r in P.items()}
  rh_r = {p: r["real_held"]["rhythm_hz_median"] for p, r in P.items()}
  tp_s = {p: r["sim_K_effort"]["qturns_per_cycle"] for p, r in P.items()}
  tp_r = {p: r["real_held"]["qturns_per_cycle"] for p, r in P.items()}
  t, cyc, tc, qc = trial1_series()
  hi_r = max(list(rh_s.values()) + list(rh_r.values()))
  hi_t = max(list(tp_s.values()) + list(tp_r.values()))
  n_runs = ", ".join(f"{p} {P[p]['real_held']['n_runs']}" for p in POL if p in P)
  for name, th in MRF.THEMES.items():
    MRF.style(th)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7), gridspec_kw=dict(width_ratios=[1, 1, 1.25], wspace=0.5))
    lo_r = min(list(rh_s.values()) + list(rh_r.values()))
    slope_panel(axes[0], th, rh_s, rh_r, lambda v: f"{v:.2f}", (lo_r - 0.08, hi_r + 0.06),
                "Finger rhythm", "finger cycles per second (Hz)")
    slope_panel(axes[1], th, tp_s, tp_r, lambda v: f"{v:.2f}", (-0.05, hi_t * 1.15),
                "Quarter turns per finger cycle", "cube quarter turns per finger cycle")
    ax = axes[2]
    ax.plot(t, cyc, color=th["ink2"], lw=2, ls=(0, (3, 2)), label="finger cycles (joint encoders)")
    ax.plot(tc, qc, color=th["C5"], lw=2, label="cube quarter turns (camera)")
    ax.text(t[-1] + 1, cyc[-1], f"{cyc[-1]:.1f} cycles", va="center", fontsize=9, color=th["ink"])
    ax.text(tc[-1] + 1, qc[-1] - 0.9, f"{qc[-1]:.1f} q-turns", va="center", fontsize=9, color=th["ink"])
    ax.set_xlim(0, 75)
    ax.set_xticks([0, 15, 30, 45, 60])
    ax.set_xlabel("seconds from policy start", fontsize=9)
    ax.set_ylabel("cumulative count", fontsize=9)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_title("25 Nov trial 1 (C5 seed 3)")
    ax.legend(loc="upper left", fontsize=8.5, handlelength=2.2)
    ax.text(0.98, 0.04, f"slope {cam1['slope_qturns_per_cycle']:.2f} q-turns/cycle, r = {cam1['r']:.3f}",
            transform=ax.transAxes, ha="right", fontsize=8.5, color=th["ink2"])
    fig.text(0.01, 0.015,
             "Simulation: kit-start cache, 350 mA-equivalent actuator, seeds 9101+9102, environments held >= 20 s (pre-registered). "
             f"Real hand: runs held >= 20 s, 3 s to the drop (post hoc; runs: {n_runs}). R1 not drawn: no run held >= 20 s.",
             fontsize=7.3, color=th["muted"])
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.2)
    suffix = "" if name == "light" else "_dark"
    (E / "figures").mkdir(exist_ok=True)
    fig.savefig(E / "figures" / f"gait_efficiency{suffix}.png", dpi=150)
    plt.close(fig)
  print("wrote", E / "figures")


if __name__ == "__main__":
  main()
