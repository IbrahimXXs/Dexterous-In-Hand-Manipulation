#!/usr/bin/env python3
"""Seed jury, Step 2 figure (light + dark), README style (scripts/make_readme_figures.py THEMES/style unchanged).

Left: setting K, survival at 60 s vs turns per 30 s per policy; the non-inferiority region (PLAN 2.4) is shaded.
Middle: stress survival (mean of the three presets) minus C5s3's, with the bootstrap 95% CI; +0.03 threshold marked.
Right: survival at 60 s per condition and policy.

    uv run --no-project --with matplotlib --with numpy \
        python training_source/project/experiments/seed_jury/scripts/figure_sim.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

E = Path(__file__).resolve().parents[1]
ROOT = E.parents[3]


def _load(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[name] = mod
  spec.loader.exec_module(mod)
  return mod


MRF = _load("make_readme_figures", ROOT / "scripts" / "make_readme_figures.py")
plt = MRF.plt
CONDS = ("K", "A", "heavy_slippery", "high_delay", "obs_noise_x3")
# policy -> (theme colour key, marker, filled); identity never by colour alone (marker + legend)
POL = {"J-median": ("A", "o", True), "J-mean": ("RET", "s", True), "J-veto": ("ink", "D", True),
       "C5s1": ("muted", "v", False), "C5s2": ("muted", "^", False), "C5s3": ("C5", "*", True)}


def mk(ax, th, p, x, y, s=60, z=3):
  col, m, filled = POL[p]
  ax.scatter([x], [y], s=s * (3.2 if m == "*" else 1), marker=m, color=th[col] if filled else th["surface"],
             edgecolor=th[col] if not filled else th["surface"], linewidth=1.6 if not filled else 1.2, zorder=z)


def main() -> None:
  o = json.loads((E / "results" / "sim.json").read_text())
  C, cr, ref = o["cells"], o["criteria"], o["reference_C5s3"]
  for name, th in MRF.THEMES.items():
    MRF.style(th)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9), gridspec_kw=dict(width_ratios=[1, 1, 1.35], wspace=0.38))
    # left: non-inferiority at K
    ax = axes[0]
    xs = [C[f"K/{p}"]["survival_60s"] for p in POL]
    ys = [C[f"K/{p}"]["turns_per_30s"] for p in POL]
    x0, y0 = ref["K_survival_60s"] - 0.01, 0.9 * ref["K_turns_per_30s"]
    lo_x, hi_x = min(xs + [x0]) - 0.03, max(xs) + 0.03
    lo_y, hi_y = min(ys + [y0]) * 0.9, max(ys) * 1.08
    ax.fill_between([x0, hi_x], y0, hi_y, color=th["grid"], lw=0, zorder=0)
    ax.text(hi_x - 0.004, hi_y - (hi_y - lo_y) * 0.03, "non-inferior to C5s3", ha="right", va="top", fontsize=8.5, color=th["ink2"])
    for p in POL:
      mk(ax, th, p, C[f"K/{p}"]["survival_60s"], C[f"K/{p}"]["turns_per_30s"])
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(lo_y, hi_y)
    ax.set_xlabel("survival at 60 s", fontsize=9)
    ax.set_ylabel("turns per 30 s", fontsize=9)
    ax.set_title("Kit starts, 350 mA-equivalent actuator")
    # middle: stress survival difference vs C5s3
    ax = axes[1]
    order = ["J-median", "J-mean", "J-veto"]
    for i, p in enumerate(order):
      d, ci = cr[p]["stress_delta"], cr[p]["stress_delta_ci95"]
      ax.plot(ci, [i, i], color=th[POL[p][0]], lw=2, zorder=2)
      mk(ax, th, p, d, i)
      ax.text(max(ci[1], d) + 0.004, i + 0.22, f"{d:+.3f}", fontsize=8.5, color=th["ink"])
    ax.axvline(0, color=th["axis"], lw=1, zorder=1)
    ax.axvline(0.03, color=th["ink2"], lw=1, ls=(0, (3, 2)), zorder=1)
    ax.text(0.031, len(order) - 0.45, "+0.03 needed", fontsize=8, color=th["ink2"])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.set_ylim(-0.6, len(order) - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel("stress survival at 60 s minus C5s3 (95% CI)", fontsize=9)
    ax.set_title("Primary: three stress presets")
    # right: survival per condition
    ax = axes[2]
    off = {p: (i - 2.5) * 0.12 for i, p in enumerate(POL)}
    for j, c in enumerate(CONDS):
      for p in POL:
        mk(ax, th, p, j + off[p], C[f"{c}/{p}"]["survival_60s"], s=40)
    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels(["K", "A", "heavy\nslippery", "high\ndelay", "obs\nnoise ×3"], fontsize=8.5)
    ax.set_ylabel("survival at 60 s", fontsize=9)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_title("Survival at 60 s by condition")
    hs = [plt.Line2D([], [], ls="", marker=POL[p][1], markersize=8 if POL[p][1] != "*" else 12,
                     markerfacecolor=th[POL[p][0]] if POL[p][2] else th["surface"], markeredgecolor=th[POL[p][0]]) for p in POL]
    fig.legend(hs, list(POL), loc="upper center", ncol=6, fontsize=9, bbox_to_anchor=(0.5, 0.995), handletextpad=0.3, columnspacing=1.4)
    sel = o["selected"] or "none"
    fig.text(0.01, 0.015, f"Pre-registered (PLAN.md 2.4). Seeds 9401+9402, 512 environments per cell; J-veto τ from calibration seed 9411. "
             f"Selected variant: {sel}.", fontsize=7.5, color=th["muted"])
    fig.subplots_adjust(left=0.06, right=0.985, top=0.83, bottom=0.17)
    suffix = "" if name == "light" else "_dark"
    (E / "figures").mkdir(exist_ok=True)
    fig.savefig(E / "figures" / f"sim{suffix}.png", dpi=150)
    plt.close(fig)
  print("wrote", E / "figures")


if __name__ == "__main__":
  main()
