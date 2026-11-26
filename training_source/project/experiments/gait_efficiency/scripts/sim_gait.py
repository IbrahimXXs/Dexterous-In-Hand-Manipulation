#!/usr/bin/env python3
"""Gait efficiency, simulation half (pre-registered in ANALYSIS_PLAN.md; simulation only, no hardware, no training).

Rollouts reuse the existing harness unchanged: ``c5_hw_v1/scripts/eval_items.py::rollout`` (GRV2 ``make_env``:
assignable reset, frozen penalty weights, first episode only, timeout beyond the horizon) under
``c5_hw_v2/scripts/eval_items.py::setup`` (setting K = the kit-start cache; actuator tasks "default" and "effort").
This script only wraps the environment constructor, as ``c5_hw_v2`` does, to
  * record the measured joint positions ``robot.data.joint_pos`` after every control step (policy rate), and
  * run model_400 with the action squash it was trained with ("clip"; the study tasks use "tanh").
Cube yaw is the rollout's own ``yaw_inc`` (genuine yaw increment while alive; unwrapped yaw = cumulative sum).

    python experiments/gait_efficiency/scripts/sim_gait.py --make-items
    python experiments/gait_efficiency/scripts/sim_gait.py --items experiments/gait_efficiency/items/sim_items.json
    python experiments/gait_efficiency/scripts/sim_gait.py --analyse
Writes data/<key>.npz (git-ignored), results/items/<key>.json and results/sim.json.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
E = PROJECT / "experiments" / "gait_efficiency"
ROOT = PROJECT.parents[1]


def _load(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[name] = mod
  spec.loader.exec_module(mod)
  return mod


G = _load("gait", E / "scripts" / "gait.py")

# Same checkpoints as on the hand (sha256 equal to LEAP_Hand_RL_Deploy_Kit/policies/<kit name>/model_*.pt).
POLICIES = {
  "M400": ("checkpoint/model_400.pt", "M400"),
  "A3": ("training_source/project/experiments/hwlimit_ac/checkpoints/A_seed3_selected_model_2600.pt", "A3"),
  "R1": ("training_source/project/experiments/retention_v1/checkpoints/R1_selected_model_2600.pt", "R1"),
  "P0": ("training_source/project/experiments/grasp_rotation_v2/checkpoints/P0_1_selected_model_1500.pt", "P0"),
  "C5s1": ("training_source/project/experiments/grasp_rotation_v2/checkpoints/C5_1_selected_model_2400.pt", "C5"),
  "C5s3": ("training_source/project/experiments/grasp_rotation_v2/checkpoints/C5_3_selected_model_3000.pt", "C5s3"),
  # optional extras (not run on the hand; simulation only)
  "EFFORT_1": ("training_source/project/experiments/c5_hw_v1/checkpoints/EFFORT_1_selected_model_2900.pt", None),
  "C5HW2_1": ("training_source/project/experiments/c5_hw_v2/checkpoints/C5HW2_1_selected_model_3000.pt", None),
}
SQUASH = {"M400": "clip"}  # model_400's kit runtime and training task clip; all study policies use tanh
SETTINGS = (("K", "effort"), ("A", "default"))  # primary; reference
SEEDS = (9101, 9102)
NUM_ENVS, STEPS = 256, 1200
MIN_HELD_S = 20.0
NBOOT = 10000


def items() -> list[dict]:
  return [{"key": f"{st}_{pol}_{act}_s{seed}", "setting": st, "actuator": act, "policy": pol, "seed": seed,
           "num_envs": NUM_ENVS, "steps": STEPS}
          for st, act in SETTINGS for pol in POLICIES for seed in SEEDS]


def sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def check_kit_identity() -> dict:
  out = {}
  for pol, (ck, kit) in POLICIES.items():
    h = sha(ROOT / ck)
    if kit is not None:
      k = next((ROOT / "LEAP_Hand_RL_Deploy_Kit" / "policies" / kit).glob("model_*.pt"))
      assert sha(k) == h, (pol, ck, k)
    out[pol] = {"checkpoint": ck, "sha256": h, "kit_policy": kit}
  return out


@contextlib.contextmanager
def recording(policy: str, store: dict):
  """Wrap GRV2's make_env (already wrapped by c5_hw_v2's setup for setting K): record joint positions after each
  control step and set model_400's action squash.  Nothing on disk changes."""
  from in_hand_rotation_mjlab.grasp_rotation_v2 import rollout as R

  inner = R.make_env

  def make_env(task, *a, **k):
    env, raw, agent_cfg = inner(task, *a, **k)
    term = raw.action_manager.get_term("joint_pos")
    if policy in SQUASH:
      assert hasattr(term.cfg, "squash"), type(term.cfg)
      term.cfg.squash = SQUASH[policy]
    store["squash"] = getattr(term.cfg, "squash", None)
    robot = raw.scene["robot"]
    store["joint_names"] = list(robot.joint_names)
    q = store["q"] = []
    step = env.step

    def recorded_step(actions):
      out = step(actions)
      q.append(robot.data.joint_pos.detach().float().cpu().numpy())
      return out

    env.step = recorded_step
    return env, raw, agent_cfg

  R.make_env = make_env
  try:
    yield
  finally:
    R.make_env = inner


def run_items(path: str, device: str) -> None:
  V2 = _load("c5_hw_v2_eval_items", PROJECT / "experiments" / "c5_hw_v2" / "scripts" / "eval_items.py")
  V1 = V2.V1
  ident = check_kit_identity()
  (E / "results" / "items" / ".claims").mkdir(parents=True, exist_ok=True)
  (E / "data").mkdir(parents=True, exist_ok=True)
  for it in json.loads(Path(path).read_text()):
    key = it["key"]
    done = E / "results" / "items" / f"{key}.json"
    claim = E / "results" / "items" / ".claims" / key
    if done.exists():
      continue
    try:
      os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
      continue
    t0 = time.time()
    pol, st, act = it["policy"], it["setting"], it["actuator"]
    ck = str(ROOT / POLICIES[pol][0])
    store: dict = {}
    with V2.setup(act, st), recording(pol, store):
      assign = V2.assignment(st, V1.SETTINGS.get(st), it["num_envs"])
      tr = V1.rollout(V2.ACTUATOR_TASK[act], ck, it["seed"], it["num_envs"], it["steps"], device, "nominal", assign)
    q = np.stack(store["q"], axis=0).astype(np.float32)  # (T, n, 16), sim joint order
    np.savez_compressed(E / "data" / f"{key}.npz", q=q, yaw_inc=tr["yaw_inc"], end_step=tr["end_step"],
                        end_code=tr["end_code"], cache_id=tr["cache_id"], dt=tr["dt"], joint_names=np.array(store["joint_names"]))
    alive = np.where(tr["end_step"] < 0, it["steps"], tr["end_step"]) * float(tr["dt"])
    rec = {**it, "task": V2.ACTUATOR_TASK[act], "checkpoint": POLICIES[pol][0], "sha256": ident[pol]["sha256"],
           "squash": store["squash"], "seconds": round(time.time() - t0, 1), "survival_60s": float(np.mean(alive >= it["steps"] * float(tr["dt"]) - 1e-6)),
           "held_s_mean": float(alive.mean())}
    done.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"[{key}] squash {store['squash']} survival60 {rec['survival_60s']:.3f} ({rec['seconds']} s)", flush=True)


def per_env(key: str, band=G.BAND_HZ) -> list[dict]:
  d = np.load(E / "data" / f"{key}.npz")
  q, yaw, end, dt = d["q"], d["yaw_inc"].astype(np.float64), d["end_step"], float(d["dt"])
  T, n, _ = q.shape
  t = np.arange(T) * dt  # sample k: joint positions after control step k, k * dt from the policy start
  rows = []
  for e in range(n):
    last = T - 1 if end[e] < 0 else int(end[e]) - 1  # last sample while still held (the terminal step is excluded)
    held_s = (last + 1) * dt
    c = G.count_cycles(q[:, e, :], t, t_end_s=last * dt, band=band)
    qt_all = 4.0 * abs(yaw[: last + 1, e].sum()) / (2 * np.pi)  # quarter turns over the whole held time (as the operator counts)
    i3 = int(round(G.T_START_S / dt))
    qt_win = 4.0 * abs(yaw[i3: last + 1, e].sum()) / (2 * np.pi)  # sensitivity: only the counting window
    rows.append({"held_s": held_s, "qturns": qt_all, "qturns_window": qt_win, "status": c["status"], "cycles": c.get("cycles"),
                 "window_s": c.get("window_s"), "rhythm_hz": c.get("rhythm_hz"),
                 "stroke_amplitude_median_rad": c.get("stroke_amplitude_median_rad"),
                 "strokes": c.get("stroke_amplitudes_rad", [])})
  return rows


def summarise(rows: list[dict], rng, *, min_held_s: float = MIN_HELD_S, qkey: str = "qturns") -> dict:
  rs = [r for r in rows if r["status"] == "ok" and r["held_s"] >= min_held_s]
  if not rs:
    return {"n_envs": 0}
  qt = np.array([r[qkey] for r in rs])
  cy = np.array([r["cycles"] for r in rs])
  ws = np.array([r["window_s"] for r in rs])
  bs = np.empty(NBOOT)
  for i in range(NBOOT):
    j = rng.integers(0, len(rs), len(rs))
    bs[i] = qt[j].sum() / cy[j].sum()
  strokes = np.concatenate([np.asarray(r["strokes"]) for r in rs]) if rs else np.array([])
  return {"n_envs": len(rs), "qturns_per_cycle": float(qt.sum() / cy.sum()),
          "qturns_per_cycle_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
          "rhythm_hz_median": float(np.median([r["rhythm_hz"] for r in rs])),
          "cycle_rate_hz": float(cy.sum() / ws.sum()), "qturns_per_s": float(qt.sum() / ws.sum()),
          "stroke_amplitude_median_rad": float(np.median(strokes)) if strokes.size else float("nan")}


def analyse() -> None:
  rng = np.random.default_rng(20261128)
  out = {"plan": "ANALYSIS_PLAN.md", "settings": {}, "checkpoints": check_kit_identity()}
  for st, act in SETTINGS:
    S = out["settings"][f"{st}/{act}"] = {}
    for pol in POLICIES:
      rows = [r for seed in SEEDS for r in per_env(f"{st}_{pol}_{act}_s{seed}")]
      held = np.array([r["held_s"] for r in rows])
      S[pol] = {"n_envs_total": len(rows), "survival_60s": float(np.mean(held >= STEPS * 0.05 - 1e-6)),
                "held_ge_20s_share": float(np.mean(held >= MIN_HELD_S)),
                "primary": summarise(rows, rng),
                "all_envs_ge_10s": summarise(rows, rng, min_held_s=10.0),
                "qturns_window_only": summarise(rows, rng, qkey="qturns_window")}
  (E / "results" / "sim.json").write_text(json.dumps(out, indent=1) + "\n")


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--make-items", action="store_true")
  ap.add_argument("--items")
  ap.add_argument("--analyse", action="store_true")
  ap.add_argument("--device", default="cuda:0")
  a = ap.parse_args()
  if a.make_items:
    (E / "items").mkdir(parents=True, exist_ok=True)
    (E / "items" / "sim_items.json").write_text(json.dumps(items(), indent=1) + "\n")
    print("wrote", len(items()), "items")
  if a.items:
    run_items(a.items, a.device)
  if a.analyse:
    analyse()


if __name__ == "__main__":
  main()
