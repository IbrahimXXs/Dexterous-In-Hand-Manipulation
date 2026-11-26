#!/usr/bin/env python3
"""Real-hand policy comparison: schedule, observer record, analysis.

Policies (kit shortcuts in policies/):
  A3  baseline A, seed 3, trained inside the kit limits          (hwlimit_ac)
  R1  A + retention reward                                       (retention_v1)
  P0  R1 without the position part of the retention penalty      (grasp_rotation_v2)
  C5  R1 with the yaw-reward clip raised to +-0.5 rad/s          (grasp_rotation_v2)

Questions (pre-specified before the hardware runs):
  Q1  A3 vs C5  which rotates faster?                    (observed turns per minute)
  Q2  R1 vs P0  which keeps the cube more reliably?      (drops, time held)
  Q3  A3 vs R1  did the retention reward improve real-world stability?   (drops, time held)
  Q4  C5 vs P0  speed vs retention trade-off              (both)

Design: blocks of the 4 policies in a balanced Latin-square order (each policy
appears once in every position and follows every other policy once per 4
blocks), so warm-up, motor heating and the observer's placement practice do not
favour one policy.  Every trial is one powered run of the same length.

    trials.py plan  [--blocks 5]          print the schedule
    trials.py next  [--blocks 5]          name of the next scheduled policy
    trials.py record --policy C5 --log logs/physical_X.npz   (asks the observer)
    trials.py compare                      tables for Q1-Q4 -> logs/trial_comparison.md/.json

Plans (environment variable LEAP_TRIAL_PLAN; use a separate LEAP_TRIALS_CSV per session):
  nov20    (default) the 20 Nov design above
  c5seeds  24 Nov: C5 (seed 1, tested 20 Nov) vs C5s3, blocks C5s3 C5 C5 C5s3 (ABBA), 10 attempts each;
           C5s2 optional extra.  Q5: C5 vs C5s3, Q6: C5 vs C5s2 - safety stops, failures, rotations.
  final3   DRAFT (docs/hardware_protocol_final_DRAFT.md, not approved): M400 vs A3 vs C5s3, balanced Williams
           Latin square of six 3-attempt sequences; TRIAL_BLOCKS=12 -> 36 attempts, 12 per arm.
           Q7: M400 vs C5s3, Q8: A3 vs C5s3.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parents[1]
CSV = Path(os.environ.get("LEAP_TRIALS_CSV", KIT / "logs" / "trials.csv"))  # override for tests
POLICIES = ("A3", "C5", "R1", "P0")
EXTRA_POLICIES = ("M400", "C5s2", "C5s3")  # outside the 20 Nov schedule: model_400 reference; C5 seeds 2 and 3
# balanced (Williams) Latin square for 4 conditions
SQUARE = (("A3", "C5", "R1", "P0"), ("C5", "P0", "A3", "R1"), ("R1", "A3", "P0", "C5"), ("P0", "R1", "C5", "A3"))
QUESTIONS = (
    ("Q1", "A3", "C5", "which rotates faster?", "speed"),
    ("Q2", "R1", "P0", "which keeps the cube more reliably?", "retention"),
    ("Q3", "A3", "R1", "did the retention reward improve real-world stability?", "retention"),
    ("Q4", "C5", "P0", "speed vs retention trade-off", "both"),
)
QUESTIONS_C5 = (
    ("Q5", "C5", "C5s3", "does C5 seed 3 trip the guards less, at what speed?", "guard"),
    ("Q6", "C5", "C5s2", "does C5 seed 2 trip the guards less, at what speed?", "guard"),
)
QUESTIONS_FINAL = (
    ("Q7", "M400", "C5s3", "does C5s3 beat the course baseline?", "both"),
    ("Q8", "A3", "C5s3", "does C5s3 beat the PPO baseline?", "both"),
)
# balanced (Williams) Latin square for 3 conditions: every arm equally often in each position and after each other arm
SQUARE_FINAL3 = (("M400", "A3", "C5s3"), ("A3", "C5s3", "M400"), ("C5s3", "M400", "A3"),
                 ("M400", "C5s3", "A3"), ("A3", "M400", "C5s3"), ("C5s3", "A3", "M400"))
PLANS = {
    "nov20": (POLICIES, SQUARE, QUESTIONS),
    "c5seeds": (("C5", "C5s3"), (("C5s3", "C5", "C5", "C5s3"),), QUESTIONS_C5),
    "final3": (("M400", "A3", "C5s3"), SQUARE_FINAL3, QUESTIONS_FINAL),
}
PLAN = os.environ.get("LEAP_TRIAL_PLAN", "nov20")
if PLAN not in PLANS:
    sys.exit(f"unknown LEAP_TRIAL_PLAN {PLAN!r} ({', '.join(PLANS)})")
PLAN_POLICIES, PLAN_BLOCKS, PLAN_QUESTIONS = PLANS[PLAN]
# Simulation reference (held-out, 30 s, full grasp distribution; READMEs of hwlimit_ac / grasp_rotation_v2).
# Sim "failure" includes the tilt/drift termination rule, not only real drops.
SIM_REF = {"A3": (1.329, 0.240), "R1": (1.158, 0.190), "P0": (1.273, 0.115), "C5": (1.869, 0.199),
           "M400": (float("nan"), float("nan")),
           "C5s2": (1.826, 0.154), "C5s3": (1.685, 0.200)}  # grasp_rotation_v2/results/c5_seeds.md  # model_400 was trained with wider limits; no comparable sim number
FIELDS = ("trial", "time", "policy", "policy_dir", "checkpoint_sha256", "log", "run_s", "completed", "stop_reason",
          "observed_turns", "dropped", "drop_time_s", "intervened", "notes",
          "est_turns", "max_current_ma", "max_temp_c", "jitter_frac_gt_60ms")


def schedule(blocks: int) -> list[str]:
    return [p for b in range(blocks) for p in PLAN_BLOCKS[b % len(PLAN_BLOCKS)]]


def read_rows() -> list[dict]:
    if not CSV.exists():
        return []
    with open(CSV, newline="") as f:
        return list(csv.DictReader(f))


def scheduled_done() -> int:
    return sum(r["policy"] in PLAN_POLICIES for r in read_rows())


def cmd_plan(a) -> None:
    done = scheduled_done()
    for i, p in enumerate(schedule(a.blocks)):
        mark = "done" if i < done else ("NEXT" if i == done else "")
        print(f"{i + 1:3d}  block {i // len(PLAN_BLOCKS[0]) + 1}  {p}  {mark}")


def cmd_next(a) -> None:
    sch = schedule(a.blocks)
    done = scheduled_done()
    if done >= len(sch):
        print("DONE")
        return
    print(sch[done])


def _ask(prompt: str, kind: str = "str", allow_blank: bool = True):
    while True:
        try:
            s = input(prompt).strip()
        except EOFError:  # no more input: treat as blank / "no"
            s = ""
            allow_blank = True
        if s == "" and allow_blank:
            return None
        if kind == "float":
            try:
                return float(s)
            except ValueError:
                print("   please type a number (or leave blank)")
                continue
        if kind == "yn":
            if s.lower() in ("y", "yes"):
                return True
            if s.lower() in ("n", "no", ""):
                return False
            print("   please type y or n")
            continue
        return s


def cmd_record(a) -> None:
    d = np.load(a.log, allow_pickle=False)
    meta = json.loads(str(d["run_meta_json"])) if "run_meta_json" in d.files else {}
    t = d["elapsed_s"]
    run_s = float(t[-1] - t[0]) if len(t) else 0.0
    est = float(d["mon_est_yaw_rad"][-1]) / (2 * math.pi) if "mon_est_yaw_rad" in d.files and len(t) else float("nan")
    jit = float(np.mean(np.diff(t) > 0.060)) if len(t) > 1 else float("nan")
    temp = d["temperature_c"]
    stop = str(d["stop_reason"])
    print(f"\n=== Trial record: {a.policy}  ({run_s:.1f} s, stop: {stop}, estimated turns {est:+.2f}) ===")
    if stop.startswith("error"):
        print(f"  NOTE: this run ended early by a SAFETY STOP after {run_s:.1f} s. It is recorded as a failure of\n"
              f"  {a.policy} (do not repeat it to get a 'better' run); answer the questions for the part that ran.")
    print("Answer from what YOU saw on the real cube (Enter = not sure / no).")
    turns = _ask("  Turns the cube made in the usual direction (e.g. 0.25, 1.5; negative if backwards): ", "float")
    dropped = bool(_ask("  Did the cube drop or leave the grasp? [y/N]: ", "yn", allow_blank=False) or False)
    drop_t = _ask("  About when, in seconds after the policy started? ", "float") if dropped else None
    inter = bool(_ask("  Did anyone touch / help the cube during the run? [y/N]: ", "yn", allow_blank=False) or False)
    notes = _ask("  Notes (optional): ") or ""
    rows = read_rows()
    row = {
        "trial": len(rows) + 1, "time": datetime.now().isoformat(timespec="seconds"), "policy": a.policy,
        "policy_dir": meta.get("policy_dir", ""), "checkpoint_sha256": meta.get("policy_sha256", ""),
        "log": str(Path(a.log).name), "run_s": f"{run_s:.2f}", "completed": bool(d["completed"]),
        "stop_reason": str(d["stop_reason"]), "observed_turns": "" if turns is None else turns,
        "dropped": dropped, "drop_time_s": "" if drop_t is None else drop_t, "intervened": inter, "notes": notes,
        "est_turns": f"{est:.3f}", "max_current_ma": f"{np.abs(d['current_ma']).max():.0f}",
        "max_temp_c": f"{np.nanmax(temp):.0f}" if np.isfinite(temp).any() else "",
        "jitter_frac_gt_60ms": f"{jit:.3f}",
    }
    new = not CSV.exists()
    CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"  saved as trial {row['trial']} in {CSV}")
    if jit > 0.1:
        print(f"  WARNING: {100 * jit:.0f}% of steps took > 60 ms (USB latency timer?) - consider repeating this trial.")


# ----------------------------------------------------------------------------- analysis
def _f(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def per_trial(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        run_s = _f(r["run_s"])
        dropped = r["dropped"] == "True"
        # time the policy kept the cube: until a drop, or until a safety stop ended the run
        held_s = min(_f(r["drop_time_s"]), run_s) if dropped and not math.isnan(_f(r["drop_time_s"])) else (
            float("nan") if dropped else run_s)
        turns = _f(r["observed_turns"])
        held_for_rate = held_s if not math.isnan(held_s) else run_s
        safety = r["stop_reason"].startswith("error")
        out.append({
            "policy": r["policy"], "safety_stop": safety, "failed": dropped or safety,
            "valid": r["intervened"] != "True" and r["completed"] in ("True", "False"),
            "intervened": r["intervened"] == "True", "dropped": dropped, "held_s": held_s, "run_s": run_s,
            "turns": turns,
            # rate only when the cube was held long enough for a rate to mean something
            "turns_per_min": turns / held_for_rate * 60 if held_for_rate >= 20 and not math.isnan(turns) else float("nan"),
            "est_turns_per_min": _f(r["est_turns"]) / run_s * 60 if run_s > 1 else float("nan"),
            "jitter": _f(r["jitter_frac_gt_60ms"]),
        })
    return out


def _perm_p(x: np.ndarray, y: np.ndarray, n: int = 20000, seed: int = 0) -> float:
    """Two-sided permutation p-value for a difference in means."""
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    obs = abs(x.mean() - y.mean())
    z = np.concatenate([x, y])
    cnt = 0
    for _ in range(n):
        rng.shuffle(z)
        cnt += abs(z[: len(x)].mean() - z[len(x):].mean()) >= obs - 1e-12
    return (cnt + 1) / (n + 1)


def _boot_ci(x: np.ndarray, y: np.ndarray, n: int = 20000, seed: int = 1) -> tuple[float, float]:
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    d = [rng.choice(y, len(y)).mean() - rng.choice(x, len(x)).mean() for _ in range(n)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def summarize(tr: list[dict]) -> dict:
    s = {}
    for p in POLICIES + EXTRA_POLICIES:
        v = [t for t in tr if t["policy"] == p and not t["intervened"]]
        # primary speed: rotations counted per trial (counting stops at a drop / stop); rate only if held >= 20 s
        cnt = np.array([t["turns"] for t in v if not math.isnan(t["turns"])])
        tpm = np.array([t["turns_per_min"] for t in v if not math.isnan(t["turns_per_min"]) and not t["safety_stop"]])
        held = np.array([t["held_s"] if not math.isnan(t["held_s"]) else 0.0 for t in v])
        s[p] = {
            "n": len(v), "n_with_turns": int(len(tpm)), "n_counted": int(len(cnt)),
            "count_per_trial_mean": float(cnt.mean()) if len(cnt) else float("nan"),
            "count_per_trial_sd": float(cnt.std(ddof=1)) if len(cnt) > 1 else float("nan"),
            "turns_per_min_mean": float(tpm.mean()) if len(tpm) else float("nan"),
            "turns_per_min_sd": float(tpm.std(ddof=1)) if len(tpm) > 1 else float("nan"),
            "drops": int(sum(t["dropped"] for t in v)), "drop_rate": float(np.mean([t["dropped"] for t in v])) if v else float("nan"),
            "safety_stops": int(sum(t["safety_stop"] for t in v)),
            "failure_rate": float(np.mean([t["failed"] for t in v])) if v else float("nan"),
            "held_s_mean": float(held.mean()) if len(held) else float("nan"),
            "est_turns_per_min_mean": float(np.nanmean([t["est_turns_per_min"] for t in v])) if v else float("nan"),
            "excluded_intervened": int(sum(t["intervened"] for t in tr if t["policy"] == p)),
            "_cnt": cnt, "_tpm": tpm, "_drop": np.array([float(t["dropped"]) for t in v]), "_held": held,
            "_fail": np.array([float(t["failed"]) for t in v]),
            "_stop": np.array([float(t["safety_stop"]) for t in v]),
        }
    return s


def cmd_compare(a) -> None:
    rows = read_rows()
    if not rows:
        sys.exit("no trials recorded yet (logs/trials.csv)")
    tr = per_trial(rows)
    s = summarize(tr)
    L = [f"# Real-hand comparison ({len(rows)} trials, {datetime.now():%Y-%m-%d %H:%M})", "",
         "Observed by the operator (turns by eye, drops, interventions); trials where someone helped the cube are excluded.",
         "Small samples: treat p-values and CIs as indicative.", "",
         "Failure = cube dropped OR the run was ended by a safety guard (e.g. a finger jammed). "
         "Speed: rotations counted per trial (counting ends at a drop); rate per minute only for trials held >= 20 s.", "",
         "| Policy | trials | rotations per trial | rate while held (>= 20 s) /min | drops | safety stops | failures | mean time held [s] | est. turns/min (monitor) | sim ref: turns/30 s, failure |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for p in tuple(x for x in POLICIES + EXTRA_POLICIES if s[x]["n"]):
        v = s[p]
        sd = "" if math.isnan(v["count_per_trial_sd"]) else f" ± {v['count_per_trial_sd']:.1f}"
        L.append(f"| {p} | {v['n']} | {v['count_per_trial_mean']:.1f}{sd} (n={v['n_counted']}) | "
                 f"{v['turns_per_min_mean']:.1f} (n={v['n_with_turns']}) | {v['drops']}/{v['n']} | "
                 f"{v['safety_stops']}/{v['n']} | {100 * v['failure_rate']:.0f}% | "
                 f"{v['held_s_mean']:.1f} | {v['est_turns_per_min_mean']:+.2f} | {SIM_REF[p][0]:.2f}, {100 * SIM_REF[p][1]:.0f}% |")
    L += ["", "| Question | Comparison | Measure | First | Second | Second − first [95% CI] | p (permutation) |", "|---|---|---|---|---|---|---|"]
    out_q = {}
    for q, x, y, text, kind in PLAN_QUESTIONS:
        if kind == "guard":
            measures = [("safety-stop rate", "_stop"), ("failure rate", "_fail"), ("rotations per trial", "_cnt"),
                        ("time held [s]", "_held")]
        else:
          measures = [("rotations per trial", "_cnt")] if kind == "speed" else (
            [("failure rate", "_fail"), ("drop rate", "_drop"), ("time held [s]", "_held")] if kind == "retention" else
            [("rotations per trial", "_cnt"), ("failure rate", "_fail"), ("time held [s]", "_held")])
        out_q[q] = {}
        for name, key in measures:
            xa, ya = s[x][key], s[y][key]
            lo, hi = _boot_ci(xa, ya)
            p = _perm_p(xa, ya)
            mx = xa.mean() if len(xa) else float("nan")
            my = ya.mean() if len(ya) else float("nan")
            L.append(f"| {q}: {text} | {x} vs {y} | {name} | {mx:.2f} | {my:.2f} | {my - mx:+.2f} [{lo:+.2f}, {hi:+.2f}] | {p:.3f} |")
            out_q[q][name] = {"first": x, "second": y, "mean_first": mx, "mean_second": my, "diff": my - mx,
                              "ci95": [lo, hi], "p_perm": p, "n": [int(len(xa)), int(len(ya))]}
    jit = np.array([t["jitter"] for t in tr])
    if np.nanmax(jit) > 0.1:
        L += ["", f"Note: {int(np.sum(jit > 0.1))} trial(s) had > 10% of control steps over 60 ms (USB latency timer)."]
    md = "\n".join(L) + "\n"
    base = "trial_comparison" if CSV.name == "trials.csv" else CSV.stem + "_comparison"  # one file per session
    (CSV.parent / f"{base}.md").write_text(md)
    (CSV.parent / f"{base}.json").write_text(json.dumps(
        {"policies": {p: {k: v for k, v in s[p].items() if not k.startswith("_")} for p in POLICIES + EXTRA_POLICIES},
         "questions": out_q},
        indent=2) + "\n")
    print(md)
    print(f"-> {CSV.parent / (base + '.md')}, {CSV.parent / (base + '.json')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "next"):
        p = sub.add_parser(name)
        p.add_argument("--blocks", type=int, default=5)
    r = sub.add_parser("record")
    r.add_argument("--policy", required=True, choices=POLICIES + EXTRA_POLICIES)
    r.add_argument("--log", type=Path, required=True)
    sub.add_parser("compare")
    a = ap.parse_args()
    {"plan": cmd_plan, "next": cmd_next, "record": cmd_record, "compare": cmd_compare}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
