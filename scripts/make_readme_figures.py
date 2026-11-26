#!/usr/bin/env python3
"""Regenerate the README summary figures and headline numbers from the committed results.

Every number comes from a file already in the repository:

* real hand, 20 Nov: LEAP_Hand_RL_Deploy_Kit/audit/20261120/attempts.csv (all 37 logged attempts)
  joined with LEAP_Hand_RL_Deploy_Kit/logs/trials.csv (operator counts and drop times)
* real hand, 24 and 25 Nov: LEAP_Hand_RL_Deploy_Kit/logs/trials_20261124.csv, trials_20261125.csv
* simulation: training_source/project/experiments/grasp_rotation_v2/results/{c5_seeds,phase4}.json
  and training_source/project/experiments/c5_hw_v1/results/effort_v1_comparison.json
  and training_source/project/experiments/c5_hw_v2/results/v2_comparison.json

Outcome rules for the real hand (the same as the session reports):
* a run ended by a safety guard is a "safety stop", even if the cube had already dropped;
* otherwise a run with an operator-reported drop is "dropped";
* otherwise a run that reached 60 s is "held the whole run";
* operator stops with no recorded reason, and runs with operator intervention, are excluded.
Operator counts are quarter turns (90 deg); full revolutions = quarter turns / 4.

Usage (from the repository root):
    uv run --no-project --with matplotlib --with numpy python scripts/make_readme_figures.py
Writes docs/figures/readme/*.png (light and dark variants) and headline_numbers.json.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "LEAP_Hand_RL_Deploy_Kit"
EXP = ROOT / "training_source" / "project" / "experiments"
OUT = ROOT / "docs" / "figures" / "readme"

# ---------------------------------------------------------------- theme
# Reference data-viz palette: categorical slots 1-3 (validated all-pairs, light and dark),
# status colours for outcomes (always paired with a marker shape and a legend label).
THEMES = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7",
        A="#2a78d6", RET="#eb6834", C5="#1baf7a", REF="#898781",
        good="#0ca30c", serious="#ec835a", critical="#d03b3b",
    ),
    "dark": dict(
        surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835",
        A="#3987e5", RET="#d95926", C5="#199e70", REF="#898781",
        good="#0ca30c", serious="#ec835a", critical="#d03b3b",
    ),
}
FONT = ["Helvetica Neue", "Helvetica", "Arial", "Nimbus Sans", "DejaVu Sans"]  # Nimbus Sans: Helvetica metrics on Linux


def style(theme: dict) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT, "font.size": 10,
        "figure.facecolor": theme["surface"], "axes.facecolor": theme["surface"],
        "savefig.facecolor": theme["surface"], "text.color": theme["ink"],
        "axes.edgecolor": theme["axis"], "axes.labelcolor": theme["ink2"],
        "axes.titlecolor": theme["ink"], "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 10,
        "xtick.color": theme["muted"], "ytick.color": theme["muted"],
        "xtick.labelcolor": theme["ink2"], "ytick.labelcolor": theme["ink2"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": theme["grid"], "grid.linewidth": 0.8,
        "legend.frameon": False, "legend.labelcolor": theme["ink2"],
    })


def fnum(x: str | None) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------- real-hand data
def attempts_20nov() -> list[dict]:
    audit = list(csv.DictReader(open(KIT / "audit" / "20261120" / "attempts.csv")))
    trials = {r["trial"]: r for r in csv.DictReader(open(KIT / "logs" / "trials.csv"))}
    out = []
    for a in audit:
        row = trials.get(a["trial"]) if a["trial"] else None
        if a["category_name"] in ("operator_stop_unknown_reason", "invalid_or_incomplete"):
            continue
        dur = fnum(a["duration_s"])
        dropped = row is not None and row["dropped"] == "True"
        drop_t = fnum(row["drop_time_s"]) if row else float("nan")
        guard = bool(a["guard_type"])
        outcome = "safety stop" if guard else ("dropped" if dropped else "held")
        t_end = min(dur, drop_t) if dropped and math.isfinite(drop_t) else dur
        qt = fnum(row["observed_turns"]) if row else float("nan")
        out.append(dict(session="20 Nov", policy=a["policy"], outcome=outcome,
                        t=min(t_end, 60.0), quarter_turns=qt))
    return out


def attempts_from_trials(path: Path, session: str) -> list[dict]:
    out = []
    for r in csv.DictReader(open(path)):
        if r["intervened"] == "True":
            continue
        run_s = fnum(r["run_s"])
        dropped = r["dropped"] == "True"
        drop_t = fnum(r["drop_time_s"])
        guard = r["stop_reason"].startswith("error")
        if not guard and not dropped and r["stop_reason"] != "max_steps":
            continue  # operator stop without a drop and without a reason
        outcome = "safety stop" if guard else ("dropped" if dropped else "held")
        t_end = min(run_s, drop_t) if dropped and math.isfinite(drop_t) else run_s
        out.append(dict(session=session, policy=r["policy"], outcome=outcome,
                        t=min(t_end, 60.0), quarter_turns=fnum(r["observed_turns"])))
    return out


ROWS = [  # (label, policy, session, family)
    ("model_400 (course)", "M400", "20 Nov", "REF"),
    ("A3: PPO, real joint limits", "A3", "20 Nov", "A"),
    ("R1: + retention reward", "R1", "20 Nov", "RET"),
    ("P0: R1 without position term", "P0", "20 Nov", "RET"),
    ("C5 seed 1", "C5", "20 Nov", "C5"),
    ("C5 seed 1", "C5", "24 Nov", "C5"),
    ("C5 seed 3", "C5s3", "24 Nov", "C5"),
    ("C5 seed 3", "C5s3", "25 Nov", "C5"),
]


def real_hand_table(att: list[dict]) -> list[dict]:
    table = []
    for label, pol, ses, fam in ROWS:
        sel = [a for a in att if a["policy"] == pol and a["session"] == ses]
        qt = np.array([a["quarter_turns"] for a in sel if math.isfinite(a["quarter_turns"])])
        table.append(dict(
            label=label, policy=pol, session=ses, family=fam, attempts=len(sel),
            held=sum(a["outcome"] == "held" for a in sel),
            dropped=sum(a["outcome"] == "dropped" for a in sel),
            safety_stops=sum(a["outcome"] == "safety stop" for a in sel),
            counted_trials=int(qt.size),
            rev_per_trial_mean=float(qt.mean() / 4) if qt.size else float("nan"),
            rev_per_trial_values=[float(v / 4) for v in qt],
            time_held_s=[round(a["t"], 2) for a in sel],
            outcomes=[a["outcome"] for a in sel],
        ))
    return table


# ---------------------------------------------------------------- simulation data
def sim_tables() -> dict:
    c5 = json.load(open(EXP / "grasp_rotation_v2" / "results" / "c5_seeds.json"))
    p4 = json.load(open(EXP / "grasp_rotation_v2" / "results" / "phase4.json"))
    eff = json.load(open(EXP / "c5_hw_v1" / "results" / "effort_v1_comparison.json"))["tables"]
    A, B = c5["settingA"], c5["settingB"]["nominal_grasp_49"]
    pA, pB = p4["settingA"]["nominal"], p4["settingB"]["nominal_grasp"]
    eA, eB = eff["A/default/nominal"]["EFFORT_1"], eff["B/default/nominal"]["EFFORT_1"]
    v2 = json.load(open(EXP / "c5_hw_v2" / "results" / "v2_comparison.json"))["table"]
    hA, hB = v2["A/default/C5HW2_1"], v2["B/default/C5HW2_1"]

    def pol(name, fam, a_vals, b_vals, note="", label=None):
        return dict(name=name, family=fam, full=a_vals, grasp49=b_vals, note=note, label=label or name)

    return dict(
        policies=[
            pol("A", "A", [A[k]["turns30"] for k in ("A1", "A2", "A3")],
                [B[k]["turns30"] for k in ("A1", "A2", "A3")]),
            pol("R1", "RET", [A["R1"]["turns30"]], [B["R1"]["turns30"]]),
            pol("P0", "RET", [pA["P0"]["turns30"]], [pB["P0"]["turns30"]]),
            pol("C5", "C5", [A[k]["turns30"] for k in ("C5", "C5s2", "C5s3")],
                [B[k]["turns30"] for k in ("C5", "C5s2", "C5s3")]),
            pol("C5-EFFORT", "C5", [eA["turns_30s"]], [eB["turns_30s"]],
                note="2048 envs per eval seed, default actuator (effort_v1_comparison.json); 1 training seed",
                label="C5-EFFORT\n(1 seed)"),
            pol("C5-HW-V2", "C5", [hA["turns30"]], [hB["turns30"]],
                note="2048 envs per eval seed, default actuator (c5_hw_v2 v2_comparison.json); 1 training seed; "
                     "pre-registered verdict FAIL (single-seed exploratory)",
                label="C5-HW-V2\n(1 seed)"),
        ],
        sim_vs_real=dict(
            turns60={"A3": A["A3"]["turns60"], "R1": A["R1"]["turns60"],
                     "P0": pA["P0"]["turns60"], "C5": A["C5"]["turns60"]},
            held60={"A3": A["A3"]["held60"], "R1": A["R1"]["held60"],
                    "P0": pA["P0"]["held60"], "C5": A["C5"]["held60"]},
        ),
        inference=c5["inference"]["turns30"]["C5_vs_A_welch"],
    )


# ---------------------------------------------------------------- figures
OUTCOME_STYLE = {  # status colour + marker shape + label: never colour alone
    "held": ("good", "o", "Held the cube for the whole run"),
    "dropped": ("serious", "s", "Cube dropped"),
    "safety stop": ("critical", "D", "Safety stop (current guard)"),
}


def fig_real_hand(table: list[dict], th: dict, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.1), sharey=True,
                                   gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.12))
    rng = np.random.default_rng(0)
    y = np.arange(len(table))[::-1]
    for yi, row in zip(y, table):
        vals = np.array(row["rev_per_trial_values"])
        if vals.size:
            jitter = rng.uniform(-0.16, 0.16, vals.size)
            ax1.scatter(vals, yi + jitter, s=34, color=th[row["family"]], alpha=0.85,
                        edgecolor=th["surface"], linewidth=1.2, zorder=3)
            m = vals.mean()
            ax1.plot([m, m], [yi - 0.32, yi + 0.32], color=th["ink"], lw=2, zorder=4,
                     solid_capstyle="round")
            ax1.text(max(vals.max(), m) + 0.25, yi, f"{m:.2f}", va="center", fontsize=9,
                     color=th["ink2"])
        for t, oc in zip(row["time_held_s"], row["outcomes"]):
            ck, mk, _ = OUTCOME_STYLE[oc]
            ax2.scatter(t, yi + rng.uniform(-0.2, 0.2), s=40, marker=mk, color=th[ck],
                        edgecolor=th["surface"], linewidth=1.0, zorder=3)
        n, k = row["attempts"], row["safety_stops"]
        ax2.text(63, yi, f"{row['held']}/{n} held, {k} stop{'' if k == 1 else 's'}", va="center",
                 fontsize=8.5, color=th["ink2"])
    ax1.set_yticks(y)
    ax1.set_yticklabels([f"{r['label']}  ·  {r['session']}" for r in table])
    for ax in (ax1, ax2):
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
    ax1.set_xlim(-0.2, 6.2)
    ax1.set_xlabel("full revolutions per 60-s trial (bar = mean)")
    ax1.set_title("Rotation on the real hand")
    ax2.set_xlim(-1, 60.5)
    ax2.set_xticks([0, 10, 20, 30, 40, 50, 60])
    ax2.set_xlabel("seconds until the cube was lost or the run stopped")
    ax2.set_title("Every attempt and how it ended (60 s = whole trial)")
    handles = [plt.Line2D([], [], marker=mk, ls="", color=th[ck], markersize=7, label=lab)
               for ck, mk, lab in OUTCOME_STYLE.values()]
    ax2.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3,
               fontsize=8.5, handletextpad=0.3, columnspacing=1.2)
    fig.text(0.01, 0.005, "All logged attempts; operator stops with no recorded reason are excluded. "
             "A run ended by a safety guard counts as a safety stop even if the cube had dropped first.",
             fontsize=7.5, color=th["muted"])
    fig.subplots_adjust(left=0.235, right=0.87, top=0.9, bottom=0.21)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_sim_progress(sim: dict, th: dict, path: Path) -> None:
    pols = sim["policies"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharey=True,
                             gridspec_kw=dict(wspace=0.06))
    x = np.arange(len(pols))
    for ax, key, title in ((axes[0], "full", "Random start grasp (239 cached grasps)"),
                           (axes[1], "grasp49", "Fixed start grasp 49 (no retraining)")):
        for xi, p in zip(x, pols):
            vals = np.array(p[key])
            m = vals.mean()
            ax.bar(xi, m, width=0.62, color=th[p["family"]], alpha=0.9, zorder=2,
                   edgecolor=th["surface"], linewidth=2)
            if vals.size > 1:
                ax.scatter(np.full(vals.size, xi) + np.linspace(-0.13, 0.13, vals.size), vals,
                           s=18, color=th["ink"], zorder=3)
            ax.text(xi, max(m, vals.max()) + 0.07, f"{m:.2f}", ha="center", fontsize=9,
                    color=th["ink"])
        ax.set_xticks(x)
        ax.set_xticklabels([p["label"] for p in pols], fontsize=8.5)
        ax.tick_params(axis="x", length=0)
        ax.set_title(title)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    a_mean = np.mean(pols[0]["full"])
    axes[0].axhline(a_mean, color=th["muted"], lw=1, ls=(0, (3, 3)), zorder=1)
    axes[0].set_ylabel("held-out turns per 30 s")
    axes[0].set_ylim(0, 2.8)
    fig.text(0.01, 0.01, "Held-out seeds 6201+6202, under the real hand's joint limits. "
             "Dots = independent training seeds (A, C5: 3 seeds; others: 1). Dashed line = A mean.",
             fontsize=7.5, color=th["muted"])
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.2)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_sim_vs_real(sim: dict, real: list[dict], th: dict, path: Path) -> None:
    r20 = {r["policy"]: r for r in real if r["session"] == "20 Nov"}
    pols = [("A3", "A", "o"), ("R1", "RET", "o"), ("P0", "RET", "s"), ("C5", "C5", "o")]
    speed_real = {p: r20[p]["rev_per_trial_mean"] for p, _, _ in pols}
    held_real = {p: r20[p]["held"] / r20[p]["attempts"] for p, _, _ in pols}
    panels = [
        ("Speed: C5 fastest in both", "full revolutions in a 60-s trial",
         sim["sim_vs_real"]["turns60"], speed_real, lambda v: f"{v:.2f}", (-0.2, 4.4)),
        ("Retention: the ranking reversed", "share of trials that held the cube for 60 s",
         sim["sim_vs_real"]["held60"], held_real, lambda v: f"{v:.0%}", (-0.05, 1.0)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw=dict(wspace=0.45))
    for ax, (title, ylab, s, r, fmt, ylim) in zip(axes, panels):
        right_pos = _spread([r[p] for p, _, _ in pols], (ylim[1] - ylim[0]) * 0.055)
        left_pos = _spread([s[p] for p, _, _ in pols], (ylim[1] - ylim[0]) * 0.055)
        for (p, fam, mk), ly, ry in zip(pols, left_pos, right_pos):
            ax.plot([0, 1], [s[p], r[p]], color=th[fam], lw=2, zorder=2)
            ax.scatter([0, 1], [s[p], r[p]], s=44, marker=mk, color=th[fam], zorder=3,
                       edgecolor=th["surface"], linewidth=1.5)
            ax.text(1.07, ry, f"{p}  {fmt(r[p])}", va="center", fontsize=9, color=th["ink"])
            ax.text(-0.07, ly, f"{fmt(s[p])}  {p}", va="center", ha="right", fontsize=9,
                    color=th["ink"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(*ylim)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Simulation", "Real hand"])
        ax.tick_params(axis="x", length=0, labelsize=10)
        ax.set_yticks([])
        for sp in ("left", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.set_title(title)
        ax.text(0.5, -0.17, ylab, transform=ax.transAxes, ha="center", fontsize=9, color=th["ink2"])
    fig.text(0.01, 0.01, "Simulation: held-out, random start grasp; A3 = A training seed 3, R1/P0/C5 = training seed 1. "
             "Real hand: 20 Nov session, same policies; held = 60 s with no drop and no safety stop.",
             fontsize=7.5, color=th["muted"])
    fig.subplots_adjust(left=0.1, right=0.9, top=0.88, bottom=0.17)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _spread(vals: list[float], gap: float) -> list[float]:
    """Nudge label positions apart so direct labels never overlap."""
    order = np.argsort(vals)
    pos = np.array(vals, dtype=float)
    for _ in range(50):
        moved = False
        for i, j in zip(order[:-1], order[1:]):
            if pos[j] - pos[i] < gap:
                mid = (pos[i] + pos[j]) / 2
                pos[i], pos[j] = mid - gap / 2, mid + gap / 2
                moved = True
        if not moved:
            break
    return list(pos)


# ---------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    att = attempts_20nov()
    att += attempts_from_trials(KIT / "logs" / "trials_20261124.csv", "24 Nov")
    att += attempts_from_trials(KIT / "logs" / "trials_20261125.csv", "25 Nov")
    real = real_hand_table(att)
    sim = sim_tables()
    for name, th in THEMES.items():
        style(th)
        suffix = "" if name == "light" else "_dark"
        fig_real_hand(real, th, OUT / f"real_hand_attempts{suffix}.png")
        fig_sim_progress(sim, th, OUT / f"sim_progress{suffix}.png")
        fig_sim_vs_real(sim, real, th, OUT / f"sim_vs_real{suffix}.png")
    summary = dict(real_hand=real, simulation=sim)
    (OUT / "headline_numbers.json").write_text(json.dumps(summary, indent=1))
    for r in real:
        print(f"{r['label']:32s} {r['session']}: attempts {r['attempts']:2d}  held {r['held']}  "
              f"dropped {r['dropped']:2d}  stops {r['safety_stops']}  "
              f"rev/trial {r['rev_per_trial_mean']:.2f} (n={r['counted_trials']})")
    for p in sim["policies"]:
        print(f"sim {p['name']:10s} full {np.mean(p['full']):.3f}  grasp49 {np.mean(p['grasp49']):.3f}")


if __name__ == "__main__":
    main()
