#!/usr/bin/env python3
"""Seed jury, Step 2 (PLAN.md section 2): the three C5 seeds as one policy, in simulation (evaluation only).

The existing harness is reused unchanged: c5_hw_v1/scripts/eval_items.py ``rollout`` (GRV2 make_env, assignable
reset, effort probe, jam detector) under c5_hw_v2/scripts/eval_items.py ``setup`` (setting K = kit-start cache).
This script only
  * wraps the policy loader (``hwlimit_study.evaluation.load_study_policy``) so that the actor is a jury of the three
    seed actors, each loaded with the original loader (own normaliser), and
  * wraps the environment step to record joint positions and the joint-position command target after every step.
The environment builds the observation from its own executed command, so the "previous command" in the observation is
the jury's executed command.

    python experiments/seed_jury/scripts/sim_jury.py --make-items
    python experiments/seed_jury/scripts/sim_jury.py --items experiments/seed_jury/items/sim_items.json   # GPU worker(s)
    python experiments/seed_jury/scripts/sim_jury.py --analyse
Writes data/<key>.npz (git-ignored), results/items/<key>.json, results/tau.json, results/sim.json, results/sim.md.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
E = PROJECT / "experiments" / "seed_jury"
CK = {"C5s1": "experiments/grasp_rotation_v2/checkpoints/C5_1_selected_model_2400.pt",
      "C5s2": "experiments/grasp_rotation_v2/checkpoints/C5_2_selected_model_1500.pt",
      "C5s3": "experiments/grasp_rotation_v2/checkpoints/C5_3_selected_model_3000.pt"}
SHA = {"C5s1": "2a932dfe37a5540d", "C5s2": "49579bdf1f598d8c", "C5s3": "c221af89199cdeab"}
SEED_ORDER = ("C5s1", "C5s2", "C5s3")
VARIANTS = ("J-median", "J-mean", "J-veto", "C5s1", "C5s2", "C5s3")
JURY = ("J-median", "J-mean", "J-veto")
CONDITIONS = {"K": ("K", "effort", "nominal"), "A": ("A", "default", "nominal"),
              "heavy_slippery": ("A", "default", "heavy_slippery"), "high_delay": ("A", "default", "high_delay"),
              "obs_noise_x3": ("A", "default", "obs_noise_x3")}
STRESS = ("heavy_slippery", "high_delay", "obs_noise_x3")
EVAL_SEEDS = (9401, 9402)
CALIB = {"key": "calib_K_C5s3_s9411", "policy": "C5s3", "condition": "K", "seed": 9411, "calibration": True}
NUM_ENVS, STEPS = 256, 1200
DELTA = 1.0 / 24.0  # delta = tanh(u) * 1/24 rad for the C5 action term (delta range +-1/24, tanh squash)
SMOOTH = 10
MIN_HELD_S = 20.0
NBOOT = 10000


def _load(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[name] = mod
  spec.loader.exec_module(mod)
  return mod


def items() -> list[dict]:
  out = [dict(CALIB, num_envs=NUM_ENVS, steps=STEPS)]
  for cond in CONDITIONS:
    for v in VARIANTS:
      for s in EVAL_SEEDS:
        out.append({"key": f"{cond}_{v}_s{s}", "policy": v, "condition": cond, "seed": s, "calibration": False,
                    "num_envs": NUM_ENVS, "steps": STEPS})
  return out


# ----------------------------------------------------------------------------------------------- jury actor
class JuryActor:
  """obs -> raw action u for the environment (which applies tanh and the delta mapping).  Logs per step the
  disagreement D = ||delta(u) - median_s delta(a_s)||_2 and, for calibration, |delta(a_3) - delta(a_med)| per joint."""

  def __init__(self, actors: list, rule: str, tau=None, calibrate: bool = False):
    import torch

    self.torch, self.actors, self.rule, self.calibrate = torch, actors, rule, calibrate
    self.tau = None if tau is None else torch.as_tensor(tau, dtype=torch.float32)
    self.D, self.dev3 = [], []

  def __call__(self, obs):
    torch = self.torch
    a = torch.stack([act(obs) for act in self.actors])  # (3, n, 16) raw outputs
    t = torch.tanh(a)
    t_med = t.median(dim=0).values  # per joint, exactly one of the three (3 values)
    if self.rule in SEED_ORDER:
      u = a[SEED_ORDER.index(self.rule)]
    elif self.rule == "J-median":
      u = a.median(dim=0).values  # tanh is monotone: tanh(median a) = median tanh(a)
    elif self.rule == "J-mean":
      u = torch.atanh(t.mean(0).clamp(-1 + 1e-6, 1 - 1e-6))
    elif self.rule == "J-veto":
      tau = self.tau.to(a.device)
      veto = (t[2] - t_med).abs() * DELTA > tau
      u = torch.where(veto, a.median(dim=0).values, a[2])
    else:
      raise KeyError(self.rule)
    self.D.append(((torch.tanh(u) - t_med).norm(dim=1) * DELTA).float().cpu())
    if self.calibrate:
      self.dev3.append(((t[2] - t_med).abs() * DELTA).float().cpu())
    return u


@contextlib.contextmanager
def jury_loader(rule: str, tau, calibrate: bool, store: dict):
  import in_hand_rotation_mjlab.hwlimit_study.evaluation as HE

  orig = HE.load_study_policy

  def load(env, agent_cfg, checkpoint, device):
    actors = [orig(env, agent_cfg, str(PROJECT / CK[s]), device) for s in SEED_ORDER]
    store["jury"] = JuryActor(actors, rule, tau, calibrate)
    return store["jury"]

  HE.load_study_policy = load
  try:
    yield
  finally:
    HE.load_study_policy = orig


@contextlib.contextmanager
def recording(store: dict):
  from in_hand_rotation_mjlab.grasp_rotation_v2 import rollout as R

  inner = R.make_env

  def make_env(task, *a, **k):
    env, raw, agent_cfg = inner(task, *a, **k)
    term = raw.action_manager.get_term("joint_pos")
    store["squash"] = getattr(term.cfg, "squash", None)
    robot = raw.scene["robot"]
    store["joint_names"] = list(robot.joint_names)
    q, tgt = [], []
    store["q"], store["target"] = q, tgt
    step = env.step

    def recorded_step(actions):
      out = step(actions)
      q.append(robot.data.joint_pos.detach().float().cpu().numpy())
      tgt.append(term._target.detach().float().cpu().numpy())
      return out

    env.step = recorded_step
    return env, raw, agent_cfg

  R.make_env = make_env
  try:
    yield
  finally:
    R.make_env = inner


def run_items(path: str, device: str) -> None:
  import torch

  V2 = _load("c5_hw_v2_eval_items", PROJECT / "experiments" / "c5_hw_v2" / "scripts" / "eval_items.py")
  V1 = V2.V1
  for s, ck in CK.items():
    assert hashlib.sha256((PROJECT / ck).read_bytes()).hexdigest().startswith(SHA[s]), s
  (E / "results" / "items" / ".claims").mkdir(parents=True, exist_ok=True)
  (E / "data").mkdir(exist_ok=True)
  its = json.loads(Path(path).read_text())
  while True:
    progressed, waiting = False, False
    for it in its:
      key = it["key"]
      done = E / "results" / "items" / f"{key}.json"
      if done.exists():
        continue
      tau = None
      if it["policy"] == "J-veto":
        tp = E / "results" / "tau.json"
        if not tp.exists():
          waiting = True
          continue
        tau = json.loads(tp.read_text())["tau_rad_per_joint"]
      claim = E / "results" / "items" / ".claims" / key
      try:
        os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
      except FileExistsError:
        continue
      t0 = time.time()
      st, act, preset = CONDITIONS[it["condition"]]
      store: dict = {}
      with V2.setup(act, st), recording(store), jury_loader(it["policy"], tau, it["calibration"], store):
        tr = V1.rollout(V2.ACTUATOR_TASK[act], str(PROJECT / CK["C5s3"]), it["seed"], it["num_envs"], it["steps"], device,
                        preset, V2.assignment(st, V2.SETTINGS.get(st), it["num_envs"]))
      J = store["jury"]
      tr["D"] = torch.stack(J.D).numpy().astype(np.float32)  # (T, n)
      tr["q"] = np.stack(store["q"]).astype(np.float32)  # (T, n, 16) after each step
      tr["target"] = np.stack(store["target"]).astype(np.float32)
      if it["calibration"]:
        dev3 = torch.stack(J.dev3).numpy()  # (T, n, 16)
        T = dev3.shape[0]
        e = tr["end_step"]
        alive = np.arange(T)[:, None] <= np.where(e < 0, T, e)[None, :]
        tau_j = np.percentile(dev3[alive], 95, axis=0)
        (E / "results" / "tau.json").write_text(json.dumps({
            "rule": "tau_j = 95th percentile of |delta(a_C5s3,j) - delta(a_median,j)| over alive first-episode env-steps",
            "item": it, "n_samples": int(alive.sum()), "tau_rad_per_joint": tau_j.tolist(), "joint_names": store["joint_names"]}, indent=1) + "\n")
      np.savez_compressed(E / "data" / f"{key}.npz", **tr)
      del J, store
      import gc
      gc.collect()
      torch.cuda.empty_cache()
      summ = V1.summarise(tr)
      done.write_text(json.dumps({"item": it, "setting": st, "actuator": act, "preset": preset, "task": V2.ACTUATOR_TASK[act],
                                  "summary": summ, "host": os.uname().nodename, "seconds": round(time.time() - t0, 1)}, indent=1) + "\n")
      s = summ["60s"]
      print(f"[{key}] surv60 {s['completed']:.3f} turns30 {summ['30s']['turns']:.3f} jam {s['jam_proxy']:.4f} "
            f"({time.time() - t0:.0f}s)", flush=True)
      progressed = True
    if not progressed:
      if waiting and not (E / "results" / "tau.json").exists():
        time.sleep(30)
        continue
      break


# ----------------------------------------------------------------------------------------------- analysis
def per_env(tr: dict, V1, G) -> dict:
  from in_hand_rotation_mjlab.grasp_rotation_v2.rollout import summary_per_env

  dt, T = float(tr["dt"]), int(tr["T"])
  e = tr["end_step"]
  n = len(e)
  s30, s60 = summary_per_env(tr, 600), summary_per_env(tr, T)
  blocked = np.unpackbits(tr["blocked_idx_packed"], axis=0, count=T).astype(bool)
  jam = V1.jam_per_env(blocked, e, dt)
  # gait (as gait_efficiency/sim_gait.py per_env)
  q, yaw = tr["q"], tr["yaw_inc"].astype(np.float64)
  t = np.arange(T) * dt
  g = []
  for i in range(n):
    last = T - 1 if e[i] < 0 else int(e[i]) - 1
    held_s = (last + 1) * dt
    c = G.count_cycles(q[:, i, :], t, t_end_s=last * dt)
    g.append({"held_s": held_s, "qturns": 4.0 * abs(yaw[: last + 1, i].sum()) / (2 * np.pi), "status": c["status"],
              "cycles": c.get("cycles"), "rhythm_hz": c.get("rhythm_hz")})
  # command change and D over alive first-episode steps
  tgt = tr["target"].astype(np.float64)
  dcmd = np.abs(np.diff(tgt, axis=0)).mean(-1)  # (T-1, n): step k vs k-1
  last_pre = np.where(e < 0, T - 1, e - 1)  # last recorded target before the terminal reset
  k = np.arange(1, T)[:, None]
  m_cmd = k <= last_pre[None, :]
  D = tr["D"].astype(np.float64)
  c = np.cumsum(np.vstack([np.zeros((1, n)), D]), axis=0)
  kk = np.arange(1, T + 1)
  lo = np.maximum(0, kk - SMOOTH)
  Db = (c[kk] - c[lo]) / (kk - lo)[:, None]
  alive = np.arange(T)[:, None] <= np.where(e < 0, T, e)[None, :]
  m3 = alive & (np.arange(T)[:, None] >= int(round(3.0 / dt)))
  end_s = np.where(e < 0, T, e + 1) * dt
  m_ctl = m3 & ((np.arange(T)[:, None] * dt) <= (end_s - 5.0)[None, :])
  return {"surv30": s30["survival_s"] >= 600 * dt - 1e-6, "surv60": s60["survival_s"] >= T * dt - 1e-6,
          "turns30": s30["turns"], "jam60": jam["jam_60s"], "gait": g,
          "dcmd_sum": float(dcmd[m_cmd].sum()), "dcmd_n": int(m_cmd.sum()), "Db_alive3": Db[m3], "Db_ctl": Db[m_ctl]}


def analyse() -> None:
  V2 = _load("c5_hw_v2_eval_items", PROJECT / "experiments" / "c5_hw_v2" / "scripts" / "eval_items.py")
  V1 = V2.V1
  G = _load("gait", PROJECT / "experiments" / "gait_efficiency" / "scripts" / "gait.py")
  rng = np.random.default_rng(20261129)
  cells = {}
  for cond in CONDITIONS:
    for v in VARIANTS:
      parts = []
      for s in EVAL_SEEDS:
        d = np.load(E / "data" / f"{cond}_{v}_s{s}.npz")
        parts.append(per_env({k: d[k] for k in d.files}, V1, G))
      cells[(cond, v)] = parts
  out = {"plan": "PLAN.md section 2", "label": "PRE-REGISTERED simulation evaluation (no training); 512 environments per cell",
         "tau": json.loads((E / "results" / "tau.json").read_text()), "cells": {}}
  for (cond, v), parts in cells.items():
    cat = lambda k: np.concatenate([p[k] for p in parts])  # noqa: E731
    gs = [x for p in parts for x in p["gait"] if x["status"] == "ok" and x["held_s"] >= MIN_HELD_S]
    db = np.concatenate([p["Db_alive3"] for p in parts])
    out["cells"][f"{cond}/{v}"] = {
        "n_envs": int(cat("surv60").size), "survival_30s": float(cat("surv30").mean()), "survival_60s": float(cat("surv60").mean()),
        "turns_per_30s": float(cat("turns30").mean()), "jam_proxy": float(cat("jam60").mean()),
        "n_gait_envs": len(gs),
        "qturns_per_cycle": float(sum(x["qturns"] for x in gs) / sum(x["cycles"] for x in gs)) if gs else None,
        "rhythm_hz_median": float(np.median([x["rhythm_hz"] for x in gs])) if gs else None,
        "cmd_change_rad_per_step": float(sum(p["dcmd_sum"] for p in parts) / sum(p["dcmd_n"] for p in parts)),
        "D_mean": float(db.mean()), "D_median": float(np.median(db)), "D_p95": float(np.percentile(db, 95))}
  C = out["cells"]
  # real vs simulation disagreement (C5s3 alone, setting K, control-like steps)
  real = json.loads((E / "results" / "real_replay.json").read_text())["real_C5s3_control_Dbar"]
  simd = np.concatenate([p["Db_ctl"] for p in cells[("K", "C5s3")]])
  out["real_vs_sim_D"] = {"real": real, "sim_K_C5s3": {"n_steps": int(simd.size), "median": float(np.median(simd)), "p95": float(np.percentile(simd, 95))},
                          "ratio_median_real_over_sim": real["median"] / float(np.median(simd)),
                          "ratio_p95_real_over_sim": real["p95"] / float(np.percentile(simd, 95))}

  # success criteria (PLAN 2.4)
  def boot_diff(v, conds, key):
    b = np.zeros(NBOOT)
    for cond in conds:
      for p_v, p_r in ((cells[(cond, v)], 1.0), (cells[(cond, "C5s3")], -1.0)):
        for part in p_v:
          x = part[key].astype(np.float64)
          b += p_r * x[rng.integers(0, x.size, (NBOOT, x.size))].mean(1) / (len(conds) * len(p_v))
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

  ref = C["K/C5s3"]
  S_ref = np.mean([C[f"{c}/C5s3"]["survival_60s"] for c in STRESS])
  crit = {}
  for v in JURY:
    k = C[f"K/{v}"]
    ni = bool(k["survival_60s"] >= ref["survival_60s"] - 0.01 and k["turns_per_30s"] >= 0.90 * ref["turns_per_30s"])
    S = np.mean([C[f"{c}/{v}"]["survival_60s"] for c in STRESS])
    d = float(S - S_ref)
    ci = boot_diff(v, STRESS, "surv60")
    route1 = bool(d >= 0.03 and ci[0] > 0)
    jci = boot_diff(v, ("K",), "jam60")
    jrel = (k["jam_proxy"] - ref["jam_proxy"]) / ref["jam_proxy"] if ref["jam_proxy"] > 0 else None
    route2 = bool(ref["jam_proxy"] > 0 and k["jam_proxy"] <= 0.75 * ref["jam_proxy"] and jci[1] < 0)
    crit[v] = {"non_inferior_K": ni, "K_survival_60s": k["survival_60s"], "K_turns_per_30s": k["turns_per_30s"],
               "K_survival_margin": k["survival_60s"] - (ref["survival_60s"] - 0.01), "K_turns_ratio": k["turns_per_30s"] / ref["turns_per_30s"],
               "stress_survival_60s": float(S), "stress_delta": d, "stress_delta_ci95": ci, "route_stress": route1,
               "K_jam_rel_change": jrel, "K_jam_diff_ci95": jci, "route_jam": route2,
               "passes": bool(ni and (route1 or route2))}
  passing = [v for v in JURY if crit[v]["passes"]]
  sel = max(passing, key=lambda v: (crit[v]["stress_delta"], -(crit[v]["K_jam_rel_change"] or 0))) if passing else None
  out["reference_C5s3"] = {"K_survival_60s": ref["survival_60s"], "K_turns_per_30s": ref["turns_per_30s"], "K_jam_proxy": ref["jam_proxy"],
                           "stress_survival_60s": float(S_ref)}
  out["criteria"] = crit
  out["selected"] = sel
  (E / "results" / "sim.json").write_text(json.dumps(out, indent=1) + "\n")
  write_md(out)
  print((E / "results" / "sim.md").read_text())


def write_md(o: dict) -> None:
  C = o["cells"]
  f = lambda x, d=3: "–" if x is None else f"{x:.{d}f}"  # noqa: E731
  L = ["# Seed jury, Step 2: the jury in simulation", "",
       f"Generated by `scripts/sim_jury.py --analyse`. **{o['label']}.** Seeds 9401 + 9402, 256 environments × 60 s each; "
       "τ for J-veto from calibration seed 9411 (`results/tau.json`).", "",
       "## Survival at 60 s / turns per 30 s", "",
       "| Policy | " + " | ".join(CONDITIONS) + " |", "|---|" + "---|" * len(CONDITIONS)]
  for v in VARIANTS:
    L.append(f"| {v} | " + " | ".join(f"{f(C[f'{c}/{v}']['survival_60s'])} / {f(C[f'{c}/{v}']['turns_per_30s'],2)}" for c in CONDITIONS) + " |")
  L += ["", "## Other metrics", "",
        "| Policy / condition | Survival 30 s | Jam proxy | Rhythm Hz | Quarter turns / cycle (envs held ≥ 20 s) | Mean \\|Δcommand\\| rad/step | D mean / p95 rad |",
        "|---|---|---|---|---|---|---|"]
  for c in CONDITIONS:
    for v in VARIANTS:
      x = C[f"{c}/{v}"]
      L.append(f"| {v} / {c} | {f(x['survival_30s'])} | {f(x['jam_proxy'],4)} | {f(x['rhythm_hz_median'])} | {f(x['qturns_per_cycle'])} ({x['n_gait_envs']}) | "
               f"{f(x['cmd_change_rad_per_step'],4)} | {f(x['D_mean'],4)} / {f(x['D_p95'],4)} |")
  r = o["real_vs_sim_D"]
  L += ["", "## Pre-registered success (against C5s3 alone)", "",
        f"C5s3 reference: K survival at 60 s {f(o['reference_C5s3']['K_survival_60s'])}, turns per 30 s {f(o['reference_C5s3']['K_turns_per_30s'])}, "
        f"K jam proxy {f(o['reference_C5s3']['K_jam_proxy'],4)}, stress survival {f(o['reference_C5s3']['stress_survival_60s'])}.", "",
        "| Variant | Non-inferior at K (survival ≥ ref − 0.01; turns ≥ 0.90 × ref) | Stress Δ survival [95% CI] (≥ +0.03, CI > 0) | K jam change [CI of difference] | Passes |",
        "|---|---|---|---|---|"]
  for v, x in o["criteria"].items():
    jr = x["K_jam_rel_change"]
    jtxt = "–" if jr is None else f"{jr:+.0%}"
    L.append(f"| {v} | {'yes' if x['non_inferior_K'] else 'no'} (survival {f(x['K_survival_60s'])}, turns ratio {f(x['K_turns_ratio'])}) | "
             f"{x['stress_delta']:+.3f} [{x['stress_delta_ci95'][0]:+.3f}, {x['stress_delta_ci95'][1]:+.3f}] | "
             f"{jtxt} [{x['K_jam_diff_ci95'][0]:+.4f}, {x['K_jam_diff_ci95'][1]:+.4f}] | {'**yes**' if x['passes'] else 'no'} |")
  L += ["", f"**Selected variant: {o['selected'] or 'none'}.**" + ("" if o["selected"] else " Step 3 (hand-test preparation) is therefore not done."), "",
        "## Real vs simulated disagreement (C5s3 executing; descriptive)", "",
        f"- Real hand, control windows: median D̄ {f(r['real']['median'],4)}, p95 {f(r['real']['p95'],4)} rad ({r['real']['n_steps']} steps).",
        f"- Simulation, setting K, 3 s to end − 5 s: median {f(r['sim_K_C5s3']['median'],4)}, p95 {f(r['sim_K_C5s3']['p95'],4)} rad ({r['sim_K_C5s3']['n_steps']} steps).",
        f"- Real / simulation: median {f(r['ratio_median_real_over_sim'],2)}, p95 {f(r['ratio_p95_real_over_sim'],2)}."]
  (E / "results" / "sim.md").write_text("\n".join(L) + "\n")


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
