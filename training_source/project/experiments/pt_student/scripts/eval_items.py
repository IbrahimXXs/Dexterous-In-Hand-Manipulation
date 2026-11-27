#!/usr/bin/env python3
"""PT-student evaluation (PLAN.md section 3; simulation only, no training).

The harness is reused unchanged: c5_hw_v1/scripts/eval_items.py ``rollout`` / ``summarise`` under
c5_hw_v2/scripts/eval_items.py ``setup``.  Only the policy loader is wrapped (students; the teacher's 403-D actor
groups) and the step is wrapped to record joint positions and the command target.

    python experiments/pt_student/scripts/eval_items.py --make-items main      # S-TF, teacher, C5 seeds 1-3
    python experiments/pt_student/scripts/eval_items.py --make-items controls  # S-TF-noaux, S-MLP
    python experiments/pt_student/scripts/eval_items.py --items experiments/pt_student/items/main.json   # worker(s)
    python experiments/pt_student/scripts/eval_items.py --analyse
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
E = PROJECT / "experiments" / "pt_student"
DEPLOY = PROJECT / "experiments" / "grasp_rotation_v2" / "deploy" / "policies"
C5 = {"C5s1": ("C5_seed1/model_2400.pt", "2a932dfe37a5540d"), "C5s2": ("C5_seed2/model_1500.pt", "49579bdf1f598d8c"),
      "C5s3": ("C5_seed3/model_3000.pt", "c221af89199cdeab")}
STUDENTS = ("S-TF", "S-TF-noaux", "S-MLP")
MAIN = ("S-TF", "T", "C5s1", "C5s2", "C5s3")
CONTROLS = ("S-TF-noaux", "S-MLP")
CONDITIONS = {"K": ("K", "effort", "nominal"), "A": ("A", "default", "nominal"),
              "heavy_slippery": ("A", "default", "heavy_slippery"), "high_delay": ("A", "default", "high_delay"),
              "obs_noise_x3": ("A", "default", "obs_noise_x3")}
STRESS = ("heavy_slippery", "high_delay", "obs_noise_x3")
EVAL_SEEDS = (9501, 9502)
NUM_ENVS, STEPS = 256, 1200
MIN_HELD_S = 20.0
NBOOT = 10000


def _load(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[name] = mod
  spec.loader.exec_module(mod)
  return mod


def teacher_ck() -> str:
  c = sorted(glob.glob(str(PROJECT / "logs" / "rsl_rl" / "pt_student_teacher" / "*_pt_teacher_seed1" / "model_3000.pt")))
  assert len(c) == 1, c
  return c[0]


def checkpoint(pol: str) -> str:
  if pol in C5:
    return str(DEPLOY / C5[pol][0])
  if pol == "T":
    return teacher_ck()
  if pol in STUDENTS:
    return str(E / "runs" / pol / "final.pt")
  raise KeyError(pol)


def make_items(which: str) -> list[dict]:
  pols = MAIN if which == "main" else CONTROLS
  order = ("C5s1", "C5s2", "C5s3", "T", "S-TF", "S-TF-noaux", "S-MLP")
  return [{"key": f"{c}_{p}_s{s}", "policy": p, "condition": c, "seed": s, "num_envs": NUM_ENVS, "steps": STEPS}
          for p in order if p in pols for c in CONDITIONS for s in EVAL_SEEDS]


@contextlib.contextmanager
def policy_loader(pol: str):
  import torch

  import in_hand_rotation_mjlab.hwlimit_study.evaluation as HE

  orig = HE.load_study_policy

  def load(env, agent_cfg, ck, device):
    if pol in C5:
      return orig(env, agent_cfg, ck, device)
    if pol == "T":
      from mjlab.tasks.registry import load_rl_cfg

      return orig(env, load_rl_cfg("Mjlab-Leap-PT-Teacher"), ck, device)
    M = _load("pt_models", Path(__file__).resolve().parent / "models.py")
    m = M.load(ck, device)

    @torch.no_grad()
    def act(obs):
      return m(obs["actor"])["action"]

    return act

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
  for p, (f, h) in C5.items():
    assert hashlib.sha256((DEPLOY / f).read_bytes()).hexdigest().startswith(h), p
  (E / "results" / "items" / ".claims").mkdir(parents=True, exist_ok=True)
  (E / "data").mkdir(exist_ok=True)
  for it in json.loads(Path(path).read_text()):
    key = it["key"]
    done = E / "results" / "items" / f"{key}.json"
    if done.exists():
      continue
    try:
      ck = checkpoint(it["policy"])
    except (AssertionError, KeyError):
      continue
    if not Path(ck).exists():
      continue
    claim = E / "results" / "items" / ".claims" / key
    try:
      os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
      continue
    t0 = time.time()
    st, act, preset = CONDITIONS[it["condition"]]
    store: dict = {}
    with V2.setup(act, st), recording(store), policy_loader(it["policy"]):
      tr = V1.rollout(V2.ACTUATOR_TASK[act], ck, it["seed"], it["num_envs"], it["steps"], device, preset,
                      V2.assignment(st, V2.SETTINGS.get(st), it["num_envs"]))
    tr["q"] = np.stack(store["q"]).astype(np.float32)
    tr["target"] = np.stack(store["target"]).astype(np.float32)
    np.savez_compressed(E / "data" / f"{key}.npz", **tr)
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    summ = V1.summarise(tr)
    done.write_text(json.dumps({"item": it, "checkpoint": str(Path(ck).relative_to(PROJECT)) if str(ck).startswith(str(PROJECT)) else ck,
                                "sha256": hashlib.sha256(Path(ck).read_bytes()).hexdigest(), "setting": st, "actuator": act, "preset": preset,
                                "task": V2.ACTUATOR_TASK[act], "squash": store.get("squash"), "summary": summ,
                                "host": os.uname().nodename, "seconds": round(time.time() - t0, 1)}, indent=1) + "\n")
    print(f"[{key}] surv30 {summ['30s']['completed']:.3f} turns30 {summ['30s']['turns']:.3f} surv60 {summ['60s']['completed']:.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)


# --------------------------------------------------------------------------------------------- analysis
def per_env(tr: dict, V1, G) -> dict:
  from in_hand_rotation_mjlab.grasp_rotation_v2.rollout import summary_per_env

  dt, T = float(tr["dt"]), int(tr["T"])
  e = tr["end_step"]
  s30, s60 = summary_per_env(tr, 600), summary_per_env(tr, T)
  blocked = np.unpackbits(tr["blocked_idx_packed"], axis=0, count=T).astype(bool)
  jam = V1.jam_per_env(blocked, e, dt)
  q, yaw = tr["q"], tr["yaw_inc"].astype(np.float64)
  t = np.arange(T) * dt
  g = []
  for i in range(len(e)):
    last = T - 1 if e[i] < 0 else int(e[i]) - 1
    c = G.count_cycles(q[:, i, :], t, t_end_s=last * dt)
    g.append({"held_s": (last + 1) * dt, "qturns": 4.0 * abs(yaw[: last + 1, i].sum()) / (2 * np.pi), "status": c["status"],
              "cycles": c.get("cycles"), "rhythm_hz": c.get("rhythm_hz")})
  tgt = tr["target"].astype(np.float64)
  dcmd = np.abs(np.diff(tgt, axis=0)).mean(-1)
  last_pre = np.where(e < 0, T - 1, e - 1)
  m_cmd = np.arange(1, T)[:, None] <= last_pre[None, :]
  return {"surv30": (s30["survival_s"] >= 600 * dt - 1e-6).astype(float), "surv60": (s60["survival_s"] >= T * dt - 1e-6).astype(float),
          "turns30": s30["turns"].astype(float), "jam60": jam["jam_60s"].astype(float), "gait": g,
          "dcmd_sum": float(dcmd[m_cmd].sum()), "dcmd_n": int(m_cmd.sum())}


def analyse() -> None:
  V2 = _load("c5_hw_v2_eval_items", PROJECT / "experiments" / "c5_hw_v2" / "scripts" / "eval_items.py")
  V1 = V2.V1
  G = _load("gait", PROJECT / "experiments" / "gait_efficiency" / "scripts" / "gait.py")
  rng = np.random.default_rng(20261130)
  pols = [p for p in ("S-TF", "T", "C5s1", "C5s2", "C5s3", "S-TF-noaux", "S-MLP")
          if all((E / "data" / f"{c}_{p}_s{s}.npz").exists() for c in CONDITIONS for s in EVAL_SEEDS)]
  cells = {}
  for c in CONDITIONS:
    for p in pols:
      cells[(c, p)] = []
      for s in EVAL_SEEDS:
        d = np.load(E / "data" / f"{c}_{p}_s{s}.npz")
        cells[(c, p)].append(per_env({k: d[k] for k in d.files}, V1, G))
  out = {"plan": "PLAN.md section 3", "label": "PRE-REGISTERED simulation evaluation; S-TF, S-TF-noaux, S-MLP and the teacher are ONE training seed each; C5 rows are three training seeds",
         "policies": pols, "cells": {}}
  for (c, p), parts in cells.items():
    cat = lambda k: np.concatenate([x[k] for x in parts])  # noqa: E731
    gs = [x for part in parts for x in part["gait"] if x["status"] == "ok" and x["held_s"] >= MIN_HELD_S]
    out["cells"][f"{c}/{p}"] = {
        "n_envs": int(cat("surv60").size), "survival_30s": float(cat("surv30").mean()), "survival_60s": float(cat("surv60").mean()),
        "turns_per_30s": float(cat("turns30").mean()), "jam_proxy": float(cat("jam60").mean()), "n_gait_envs": len(gs),
        "qturns_per_cycle": float(sum(x["qturns"] for x in gs) / sum(x["cycles"] for x in gs)) if gs else None,
        "rhythm_hz_median": float(np.median([x["rhythm_hz"] for x in gs])) if gs else None,
        "cmd_change_rad_per_step": float(sum(x["dcmd_sum"] for x in parts) / sum(x["dcmd_n"] for x in parts))}
  C = out["cells"]
  c5 = ("C5s1", "C5s2", "C5s3")
  for c in CONDITIONS:
    C[f"{c}/C5mean"] = {k: float(np.mean([C[f"{c}/{p}"][k] for p in c5])) for k in
                        ("survival_30s", "survival_60s", "turns_per_30s", "jam_proxy", "cmd_change_rad_per_step")}
    for k in ("qturns_per_cycle", "rhythm_hz_median"):
      C[f"{c}/C5mean"][k] = float(np.mean([C[f"{c}/{p}"][k] for p in c5]))

  def boot_mean(parts, key):
    return np.mean([np.asarray(x[key])[rng.integers(0, len(x[key]), (NBOOT, len(x[key])))].mean(1) for x in parts], axis=0)

  def criteria(p: str) -> dict:
    tb = boot_mean(cells[("K", p)], "turns30")
    c5b = {s: boot_mean(cells[("K", s)], "turns30") for s in c5}
    ratio = tb / np.mean([c5b[s] for s in c5], axis=0)
    k, ref = C[f"K/{p}"], C["K/C5mean"]
    stress = float(np.mean([C[f"{c}/{p}"]["survival_60s"] for c in STRESS]))
    stress_ref = float(np.mean([C[f"{c}/C5mean"]["survival_60s"] for c in STRESS]))
    r = {"turns_ratio": k["turns_per_30s"] / ref["turns_per_30s"], "turns_ratio_ci95": [float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))],
         "survival_30s": k["survival_30s"], "survival_30s_ref": ref["survival_30s"], "stress_survival_60s": stress, "stress_survival_60s_ref": stress_ref}
    r["c1_turns"] = bool(r["turns_ratio"] >= 1.15 and r["turns_ratio_ci95"][0] > 1)
    r["c2_survival30"] = bool(k["survival_30s"] >= ref["survival_30s"] - 0.01)
    r["c3_stress"] = bool(stress >= stress_ref - 0.02)
    r["PASS"] = bool(r["c1_turns"] and r["c2_survival30"] and r["c3_stress"])
    per = {}
    for s in c5:
      rr = tb / c5b[s]
      per[s] = {"ratio": k["turns_per_30s"] / C[f"K/{s}"]["turns_per_30s"], "ci95": [float(np.percentile(rr, 2.5)), float(np.percentile(rr, 97.5))]}
      per[s]["beats"] = bool(per[s]["ci95"][0] > 1)
    r["vs_each_seed"] = per
    r["beats_every_seed"] = bool(all(v["beats"] for v in per.values()))
    return r

  out["criteria"] = {p: criteria(p) for p in pols if p not in c5}
  (E / "results" / "sim.json").write_text(json.dumps(out, indent=1) + "\n")
  write_md(out)
  print((E / "results" / "sim.md").read_text())


def write_md(o: dict) -> None:
  C = o["cells"]
  f = lambda x, d=3: "–" if x is None else f"{x:.{d}f}"  # noqa: E731
  rows = [p for p in o["policies"] if p not in ("C5s1", "C5s2", "C5s3")] + ["C5s1", "C5s2", "C5s3", "C5mean"]
  L = ["# PT-student: simulation evaluation", "",
       f"Generated by `scripts/eval_items.py --analyse`. **{o['label']}.** Seeds 9501 + 9502, 256 environments × 60 s each (512 per cell).", "",
       "## Turns per 30 s / survival at 30 s / survival at 60 s", "", "| Policy | " + " | ".join(CONDITIONS) + " |", "|---|" + "---|" * len(CONDITIONS)]
  for p in rows:
    L.append(f"| {p} | " + " | ".join(f"{f(C[f'{c}/{p}']['turns_per_30s'],2)} / {f(C[f'{c}/{p}']['survival_30s'])} / {f(C[f'{c}/{p}']['survival_60s'])}" for c in CONDITIONS) + " |")
  L += ["", "## Setting K (kit starts, 350 mA-equivalent actuator): other metrics", "",
        "| Policy | Jam proxy | Rhythm Hz | Quarter turns / cycle | Mean \\|Δcommand\\| rad/step |", "|---|---|---|---|---|"]
  for p in rows:
    x = C[f"K/{p}"]
    L.append(f"| {p} | {f(x['jam_proxy'],4)} | {f(x['rhythm_hz_median'])} | {f(x['qturns_per_cycle'])} | {f(x['cmd_change_rad_per_step'],4)} |")
  L += ["", "## Pre-registered criteria (vs the C5 three-seed mean, setting K)", "",
        "| Policy | Turns ratio [95% CI] (≥ 1.15, CI > 1) | Survival 30 s vs ref − 0.01 | Stress survival 60 s vs ref − 0.02 | PASS | Beats every C5 seed (ratio CI > 1) |",
        "|---|---|---|---|---|---|"]
  for p, r in o["criteria"].items():
    each = "; ".join(f"{s} {v['ratio']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]" for s, v in r["vs_each_seed"].items())
    L.append(f"| {p} | {r['turns_ratio']:.3f} [{r['turns_ratio_ci95'][0]:.3f}, {r['turns_ratio_ci95'][1]:.3f}] {'✓' if r['c1_turns'] else '✗'} | "
             f"{f(r['survival_30s'])} vs {f(r['survival_30s_ref'] - 0.01)} {'✓' if r['c2_survival30'] else '✗'} | "
             f"{f(r['stress_survival_60s'])} vs {f(r['stress_survival_60s_ref'] - 0.02)} {'✓' if r['c3_stress'] else '✗'} | "
             f"**{'PASS' if r['PASS'] else 'FAIL'}** | {'yes' if r['beats_every_seed'] else 'no'} ({each}) |")
  (E / "results" / "sim.md").write_text("\n".join(L) + "\n")


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--make-items", choices=("main", "controls"))
  ap.add_argument("--items")
  ap.add_argument("--analyse", action="store_true")
  ap.add_argument("--device", default="cuda:0")
  a = ap.parse_args()
  if a.make_items:
    (E / "items").mkdir(parents=True, exist_ok=True)
    (E / "items" / f"{a.make_items}.json").write_text(json.dumps(make_items(a.make_items), indent=1) + "\n")
    print("wrote", len(make_items(a.make_items)), "items")
  if a.items:
    run_items(a.items, a.device)
  if a.analyse:
    analyse()


if __name__ == "__main__":
  main()
