#!/usr/bin/env python3
"""PT-student figure (light + dark), README style (scripts/make_readme_figures.py THEMES/style unchanged).

One panel per condition: turns per 30 s (y) vs survival at 30 s (x), one point per policy with bootstrap 95% CIs on
both axes; the three C5 training seeds are shown individually (and their mean as a hollow marker).

    uv run --no-project --with matplotlib --with numpy \
        python training_source/project/experiments/pt_student/scripts/figure.py
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
CONDS = (("K", "Kit starts, 350 mA (primary)"), ("A", "Default starts"), ("heavy_slippery", "Heavy, slippery"),
         ("high_delay", "High delay"), ("obs_noise_x3", "Observation noise ×3"))
# policy -> (colour key, marker, filled, label)
POL = {"S-TF": ("A", "o", True, "S-TF (student, 1 seed)"), "T": ("RET", "s", True, "Teacher (privileged, 1 seed)"),
       "S-TF-noaux": ("ink", "D", True, "S-TF without aux loss"), "S-MLP": ("ink", "P", False, "S-MLP (C5 architecture)"),
       "C5s1": ("C5", "v", True, "C5 seed 1"), "C5s2": ("C5", "^", True, "C5 seed 2"), "C5s3": ("C5", "*", True, "C5 seed 3"),
       "C5mean": ("C5", "o", False, "C5 three-seed mean"),
       "T-AR": ("RET", "s", False, "Teacher + action-rate term (variant)"), "S-TF-AR": ("A", "o", False, "S-TF from that teacher (variant)")}


def main() -> None:
  o = json.loads((E / "results" / "sim.json").read_text())
  C = o["cells"]
  pols = [p for p in POL if p == "C5mean" or p in o["policies"]]
  for name, th in MRF.THEMES.items():
    MRF.style(th)
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.6), gridspec_kw=dict(wspace=0.32))
    for ax, (c, title) in zip(axes, CONDS):
      for p in pols:
        x = C[f"{c}/{p}"]
        col, mk, filled, _ = POL[p]
        if "turns_per_30s_ci95" in x:
          ax.plot(x["survival_30s_ci95"], [x["turns_per_30s"]] * 2, color=th[col], lw=1.2, alpha=0.8, zorder=2)
          ax.plot([x["survival_30s"]] * 2, x["turns_per_30s_ci95"], color=th[col], lw=1.2, alpha=0.8, zorder=2)
        ax.scatter([x["survival_30s"]], [x["turns_per_30s"]], s=110 if mk == "*" else 46, marker=mk,
                   color=th[col] if filled else th["surface"], edgecolor=th[col] if not filled else th["surface"],
                   linewidth=1.6 if not filled else 1.0, zorder=3)
      ax.set_title(title, fontsize=10)
      ax.set_xlabel("survival at 30 s", fontsize=9)
      ax.grid(True)
      ax.set_axisbelow(True)
      ax.tick_params(labelsize=8.5)
    axes[0].set_ylabel("turns per 30 s", fontsize=9)
    hs = [plt.Line2D([], [], ls="", marker=POL[p][1], markersize=11 if POL[p][1] == "*" else 7,
                     markerfacecolor=th[POL[p][0]] if POL[p][2] else th["surface"], markeredgecolor=th[POL[p][0]]) for p in pols]
    ncol = len(pols) if len(pols) <= 6 else -(-len(pols) // 2)  # two legend rows when it would not fit in one
    fig.legend(hs, [POL[p][3] for p in pols], loc="upper center", ncol=ncol, fontsize=8.5, bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.3, columnspacing=1.0)
    k = o["criteria"].get("S-TF")
    verdict = ("" if k is None else f"S-TF vs C5 mean at kit starts: turns ratio {k['turns_ratio']:.2f} [{k['turns_ratio_ci95'][0]:.2f}, "
               f"{k['turns_ratio_ci95'][1]:.2f}] (needed ≥ 1.15 with CI > 1): {'PASS' if k['PASS'] else 'FAIL'}. ")
    fig.text(0.01, 0.015, verdict + "Pre-registered (PLAN.md). Seeds 9501+9502, 512 environments per point; bars = bootstrap 95% CIs. "
             "Students and teacher: one training seed.", fontsize=7.5, color=th["muted"])
    fig.subplots_adjust(left=0.05, right=0.99, top=0.82 if ncol == len(pols) else 0.77, bottom=0.17)
    suffix = "" if name == "light" else "_dark"
    (E / "figures").mkdir(exist_ok=True)
    fig.savefig(E / "figures" / f"sim{suffix}.png", dpi=150)
    plt.close(fig)
  print("wrote", E / "figures")


if __name__ == "__main__":
  main()
