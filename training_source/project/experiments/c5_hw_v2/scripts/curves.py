#!/usr/bin/env python3
"""C5-HW-V2 validation curves and checkpoint selection: a copy of experiments/c5_hw_v1/scripts/curves.py
(itself a copy of the GRV2 / Retention-V1 curves: seed 7001, nominal, 1024 envs, 30 s, every 100 updates,
60-s points every 500; same C1-C4 criteria and selection rule: max 30-s validation turns, ties -> later)
for the single C5-HW-V2 run C5HW2_1 (PPO seed 1), validated in its own training task Mjlab-Leap-C5HW2
(kit-start resets), as C5 and C5-HW-EFFORT were in their own.  Only the run name, task and log paths differ.

    python experiments/c5_hw_v2/scripts/curves.py eval --run C5HW2_1 [--watch]
    python experiments/c5_hw_v2/scripts/curves.py assess
"""

from __future__ import annotations

import argparse
import fcntl
import glob
import os
import json
import math
import time
from pathlib import Path

import numpy as np

E = Path("experiments/c5_hw_v2")
VAL_SEED = 7001
ITERS = list(range(100, 3001, 100))
RUNS = ["C5HW2_1"]  # single-seed exploratory run (instruction 2026-11-26)
ITERS60 = list(range(500, 3001, 500))


def run_dir(run: str) -> str | None:
  c, s = run.split("_")
  d = sorted(d for d in glob.glob(f"logs/rsl_rl/c5_hw_v2/*_c5hw2_seed{s}") if "INTERRUPTED" not in d)
  return d[-1] if d else None


def cmd_eval(a):
  from in_hand_rotation_mjlab.hwlimit_study.evaluation import EvalConfig, evaluate, summarize

  out_p = E / "results" / "curves" / f"{a.run}.json"
  out_p.parent.mkdir(parents=True, exist_ok=True)
  claims = out_p.parent / ".claims"
  claims.mkdir(exist_ok=True)
  task = "Mjlab-Leap-C5HW2"

  def load():
    return json.loads(out_p.read_text()) if out_p.exists() else {"run": a.run, "points": {}}

  def claim(it: int) -> bool:
    """Several processes may evaluate the same run; each point is evaluated once (stale claims: 20 min)."""
    c = claims / f"{a.run}_{it}"
    if c.exists() and time.time() - c.stat().st_mtime > 1200:
      c.unlink(missing_ok=True)
    try:
      os.close(os.open(c, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
      return True
    except FileExistsError:
      return False

  def merge_write(d: str, it: int, point: dict):
    with open(out_p.with_suffix(".lock"), "w") as lk:
      fcntl.flock(lk, fcntl.LOCK_EX)
      res = load()
      res["run_dir"] = d
      res["points"][str(it)] = point
      tmp = out_p.with_suffix(".tmp")
      tmp.write_text(json.dumps(res, indent=1))
      tmp.replace(out_p)

  pending = [it for it in ITERS if str(it) not in load()["points"]]
  while pending:
    d = run_dir(a.run)
    ck = Path(d) / f"model_{pending[0]}.pt" if d else None
    if ck is None or not ck.exists():
      if not a.watch:
        print(f"[{a.run}] missing {ck}; stop")
        break
      time.sleep(120)
      continue
    if a.watch:
      time.sleep(30)  # let the trainer finish writing the file
    it = pending.pop(0)
    if str(it) in load()["points"] or not claim(it):
      continue
    t0 = time.time()
    pe, extra = evaluate(task, str(ck), VAL_SEED, "nominal", EvalConfig(num_envs=a.num_envs, num_steps=600, device=a.device))
    sm = summarize(pe)
    point = {k: sm[k] for k in sm} | {"cmd_at_limit_per_joint": extra["cmd_at_limit_frac_per_joint"]}
    if it in ITERS60:
      pe60, _ = evaluate(task, str(ck), VAL_SEED, "nominal", EvalConfig(num_envs=a.num_envs, num_steps=1200, device=a.device))
      s60 = summarize(pe60)
      point["long60"] = {k: s60[k] for k in ("turns", "survival_s", "failure", "completed", "mean_yaw_rate")}
    merge_write(d, it, point)
    sm = point
    print(f"[{a.run}] it {it}: turns {sm['turns']['mean']:.3f} ± {sm['turns']['ci95']:.3f}  surv {sm['survival_s']['mean']:.1f}s "
          f"fail {sm['failure']['mean']:.3f}  ({time.time() - t0:.0f}s)", flush=True)


def train_reward(d: str) -> tuple[np.ndarray, np.ndarray]:
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  ea = EventAccumulator(d, size_guidance={"scalars": 0})
  ea.Reload()
  ev = ea.Scalars("Train/mean_reward")
  return np.array([e.step for e in ev]), np.array([e.value for e in ev])


def smooth3(v: np.ndarray) -> np.ndarray:
  out = np.empty_like(v)
  for i in range(len(v)):
    out[i] = v[max(0, i - 1): i + 2].mean()
  return out


def assess_run(run: str) -> dict:
  p = E / "results" / "curves" / f"{run}.json"
  if not p.exists():
    return {"run": run, "status": "no curve"}
  res = json.loads(p.read_text())
  pts = res["points"]
  its = np.array(sorted(int(k) for k in pts))
  V = np.array([pts[str(i)]["turns"]["mean"] for i in its])
  F = np.array([pts[str(i)]["failure"]["mean"] for i in its])
  out = {"run": run, "iterations": its.tolist(), "V": V.tolist(), "V_smooth": smooth3(V).tolist(), "failure": F.tolist(),
         "complete": bool(len(its) == len(ITERS))}
  # selection: max V, ties -> later
  best = int(its[np.flatnonzero(V == V.max())[-1]])
  out["selected_iteration"] = best
  out["selected_V"] = float(V.max())
  out["final_V"] = float(V[-1]) if its[-1] == 3000 else None
  out["learning_speed_mean_V_100_3000"] = float(V.mean())
  reach = its[V >= 0.5]
  out["first_iteration_V_ge_0.5"] = int(reach[0]) if len(reach) else None
  if not out["complete"]:
    out["status"] = "incomplete"
    return out
  w = (its >= 2000)
  x, y = its[w].astype(float), V[w]
  b, a0 = np.polyfit(x, y, 1)
  resid = y - (a0 + b * x)
  se_b = math.sqrt((resid ** 2).sum() / (len(x) - 2) / ((x - x.mean()) ** 2).sum())
  t975 = 2.262  # t(0.975, df=9)
  mean_win = float(y.mean())
  w1, w2 = (its >= 2100) & (its <= 2500), (its >= 2600) & (its <= 3000)
  dV = float(V[w2].mean() - V[w1].mean())
  dF = float(F[w2].mean() - F[w1].mean())
  steps, rew = train_reward(res["run_dir"])
  r1 = rew[(steps >= 2501) & (steps <= 2750)].mean()
  r2 = rew[(steps >= 2751) & (steps <= 3000)].mean()
  dR = float((r2 - r1) / abs(r1))
  c1 = 1000 * b <= 0.05 * mean_win
  c2 = dV <= max(0.03, 0.03 * float(V[w1].mean()))
  c3 = abs(dR) <= 0.02
  c4 = abs(dF) <= 0.03
  crit = {"C1_projected_gain_per_1000": 1000 * b, "C1_threshold": 0.05 * mean_win,
          "C1_slope_ci95_per_1000": [1000 * (b - t975 * se_b), 1000 * (b + t975 * se_b)], "C1": bool(c1),
          "C2_dV_W2_minus_W1": dV, "C2_threshold": max(0.03, 0.03 * float(V[w1].mean())), "C2": bool(c2),
          "C3_reward_rel_change": dR, "C3": bool(c3), "C4_failure_change": dF, "C4": bool(c4),
          "final_window_sd_V": float(y.std(ddof=1)), "final_window_mean_V": mean_win}
  out["criteria"] = crit
  if c1 and c2 and c3 and c4:
    out["status"] = "approximately converged"
  elif (not c1 and b > 0) or (not c2 and dV > 0) or (not c3 and dR > 0) or (not c4 and dF < 0):
    out["status"] = "not converged (still improving)"
  else:
    out["status"] = "not converged (declining or unstable)"
  return out


def cmd_assess(a):
  allr = {r: assess_run(r) for r in RUNS if (E / "results" / "curves" / f"{r}.json").exists()}
  st = [v.get("status") for v in allr.values()]
  cond = {"R": {"per_seed": st, "approximately_converged": all(s == "approximately converged" for s in st)}}
  sel = {r: {"selected_iteration": v.get("selected_iteration"), "run_dir": json.loads((E / "results" / "curves" / f"{r}.json").read_text()).get("run_dir")
             if (E / "results" / "curves" / f"{r}.json").exists() else None} for r, v in allr.items()}
  out = {"runs": allr, "conditions": cond, "selection": sel, "rule": "PROTOCOL.md Sections 3-4"}
  (E / "results" / "convergence.json").write_text(json.dumps(out, indent=1))
  for r, v in allr.items():
    print(r, v.get("status"), "selected", v.get("selected_iteration"), "V_sel", v.get("selected_V"), "V_final", v.get("final_V"))
  print(cond)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  sp = p.add_subparsers(dest="cmd", required=True)
  e = sp.add_parser("eval")
  e.add_argument("--run", required=True, choices=RUNS)
  e.add_argument("--watch", action="store_true")
  e.add_argument("--num-envs", type=int, default=1024)
  e.add_argument("--device", default="cuda:0")
  sp.add_parser("assess")
  a = p.parse_args()
  {"eval": cmd_eval, "assess": cmd_assess}[a.cmd](a)


if __name__ == "__main__":
  main()
