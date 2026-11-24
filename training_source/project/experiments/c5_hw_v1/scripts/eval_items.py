#!/usr/bin/env python3
"""C5-HW-EFFORT-V1 held-out evaluation worker (PROTOCOL_EFFORT_V1.md; simulation only).

Each item = (setting, policy, actuator, preset, seed).  The rollout is GRV2's (rollout.make_env:
assignable reset, frozen penalty weights, timeout beyond the horizon, first episode only) and
additionally records, at reward time (the instant the effort penalty sees, i.e. after the
physics of the step, before any reset), per environment and actuator:

  u          |actuator force| / that environment's effort limit
  blocked    index MCP / PIP with u >= 0.98 and |command applied during the step - joint position| >= 0.2 rad
  tracking   |command - joint position| (same instant), joint speed |qd|, command at an approved limit

Saved: data/<key>.npz (per-env arrays, packed blocked flags) and results/items/<key>.json (pooled metrics).
Actuators: "default" = task Mjlab-Leap-RetV1-A (training actuator); "effort" = task
Mjlab-Leap-C5HW-EFFORT (identical except the 0.167-0.204 Nm effort range; rewards play no part).

    python experiments/c5_hw_v1/scripts/eval_items.py --make-items      # writes items/effort_v1_eval.json
    python experiments/c5_hw_v1/scripts/eval_items.py --items experiments/c5_hw_v1/items/effort_v1_eval.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

E = Path("experiments/c5_hw_v1")
G = "experiments/grasp_rotation_v2/checkpoints"
POLICIES = {"A1": "experiments/hwlimit_ac/checkpoints/A_seed1_selected_model_2600.pt",
            "A2": "experiments/hwlimit_ac/checkpoints/A_seed2_selected_model_2900.pt",
            "A3": "experiments/hwlimit_ac/checkpoints/A_seed3_selected_model_2600.pt",
            "R1": "experiments/retention_v1/checkpoints/R1_selected_model_2600.pt",
            "C5_1": f"{G}/C5_1_selected_model_2400.pt", "C5_2": f"{G}/C5_2_selected_model_1500.pt",
            "C5_3": f"{G}/C5_3_selected_model_3000.pt"}
ACTUATORS = {"default": "Mjlab-Leap-RetV1-A", "effort": "Mjlab-Leap-C5HW-EFFORT"}
SUBSET = "81,49,192,123,79,106,196,47,233,189"
SETTINGS = {"A": None, "B": "grasps:49", "C": f"grasps:{SUBSET}"}
INDEX = ("if_mcp", "if_pip")
SAT_U, JAM_ERR = 0.98, 0.2
NBINS = 1000  # utilisation histogram bins of 0.001 on [0, 1); u >= 1 in the last bin


def policy_path(p: str) -> str:
  if p in POLICIES:
    return POLICIES[p]
  if p == "EFFORT_1":
    c = sorted(glob.glob(str(E / "checkpoints" / "EFFORT_1_selected_model_*.pt")))
    assert len(c) == 1, c
    return c[0]
  raise KeyError(p)


def make_items(policies: list[str]) -> list[dict]:
  items = []
  for pol in policies:
    for act in ACTUATORS:
      for st, assign in SETTINGS.items():
        runs = [(s, "nominal") for s in (6201, 6202)]
        if st == "A":
          runs += [(6201, p) for p in ("heavy_slippery", "high_delay", "obs_noise_x3")]
        for seed, preset in runs:
          items.append({"key": f"{st}_{pol}_{act}_{preset}_s{seed}", "setting": st, "policy": pol, "actuator": act,
                        "task": ACTUATORS[act], "seed": seed, "preset": preset, "assign": assign,
                        "num_envs": 2048, "steps": 1200})
  return items


def assignment(spec, n):
  if spec is None:
    return None
  g = np.array([int(x) for x in spec.split(":", 1)[1].split(",")])
  return g[np.arange(n) % len(g)]


@torch.no_grad()
def rollout(task, ck, seed, n, T, device, preset, assign):
  from in_hand_rotation_mjlab.grasp_rotation_v2 import rollout as R
  from in_hand_rotation_mjlab.hwlimit_study.evaluation import load_study_policy
  from in_hand_rotation_mjlab.scratch_study.geometry import yaw_increment

  env, raw, agent_cfg = R.make_env(task, n, seed, device, preset=preset, num_steps=T)
  if assign is not None:
    raw._gr2_assignment = torch.as_tensor(assign, dtype=torch.long, device=device)
  actor = load_study_policy(env, agent_cfg, ck, device)
  robot, cube = raw.scene["robot"], raw.scene["cube"]
  term = raw.action_manager.get_term("joint_pos")
  lo, hi = term.approved_limits()
  ctrl = robot.indexing.ctrl_ids
  jn = list(robot.joint_names)
  an = list(robot.actuator_names) if hasattr(robot, "actuator_names") else jn
  assert an == jn, (an, jn)  # actuator order = joint order (one actuator per joint)
  ix = torch.tensor([jn.index(j) for j in INDEX], device=device)
  tm, rm = raw.termination_manager, raw.reward_manager
  # probe: a plain-function reward term evaluated every step (weight != 0), to sample at reward time
  names = list(rm._term_names)
  k = names.index("torque_penalty")
  cfg = rm._term_cfgs[k]
  assert cfg.weight != 0.0 and not hasattr(cfg.func, "reset")
  orig, snap = cfg.func, {}

  def probe(env_, **kw):
    snap["F"] = robot.data.actuator_force.clone()
    snap["L"] = raw.sim.model.actuator_forcerange[:, ctrl, 1].clone()
    snap["q"] = robot.data.joint_pos.clone()
    snap["qd"] = robot.data.joint_vel.clone()
    snap["cmd"] = term._target.clone()
    return orig(env_, **kw)

  cfg.func = probe
  env.reset()
  if assign is not None:
    assert torch.equal(raw._hand_cube_reset_cache_ids, raw._gr2_assignment)
  obs = env.get_observations()
  cache = raw._hand_cube_reset_cache_ids.clone().cpu().numpy()
  qprev = cube.data.root_link_quat_w.clone()
  first = torch.ones(n, dtype=torch.bool, device=device)
  active = torch.ones(n, dtype=torch.bool, device=device)
  end_step = torch.full((n,), -1, dtype=torch.long, device=device)
  end_code = torch.zeros(n, dtype=torch.int8, device=device)
  yaw = torch.zeros((T, n), device=device)
  blocked = torch.zeros((T, n, 2), dtype=torch.bool, device=device)
  H = {h: {"n": torch.zeros(n, device=device), "sat": torch.zeros((n, 16), device=device),
           "u": torch.zeros(16, device=device, dtype=torch.float64), "err": torch.zeros(16, device=device, dtype=torch.float64),
           "qd": torch.zeros(16, device=device, dtype=torch.float64), "lim": torch.zeros(16, device=device, dtype=torch.float64),
           "hist": torch.zeros(16 * NBINS, device=device, dtype=torch.float64)} for h in (600, T)}
  off = torch.arange(16, device=device) * NBINS
  for t in range(T):
    obs, _, dones, _ = env.step(actor(obs))
    dones = dones.bool()
    q = cube.data.root_link_quat_w
    live = active & ~dones & ~first
    yaw[t] = torch.where(live, yaw_increment(q, qprev), 0.0)
    if bool(dones.any()):
      code = torch.zeros(n, dtype=torch.int8, device=device)
      for name, c in R.TERM_CODES.items():
        if name in tm.active_terms:
          code = torch.where(tm.get_term(name).bool(), torch.full_like(code, c), code)
      newly = active & dones
      end_step = torch.where(newly, torch.full_like(end_step, t), end_step)
      end_code = torch.where(newly, code, end_code)
    # effort sample at reward time, for environments alive in their first episode (terminal step included)
    u = snap["F"].abs() / snap["L"].clamp(min=1e-9)
    err = (snap["cmd"] - snap["q"]).abs()
    sat = u >= SAT_U
    blk = sat[:, ix] & (err[:, ix] >= JAM_ERR)
    m = active
    blocked[t] = blk & m[:, None]
    atl = ((snap["cmd"] - lo).abs() < 1e-4) | ((hi - snap["cmd"]).abs() < 1e-4)
    mf = m.float()
    bins = (u.clamp(0, 0.99999) * NBINS).long() + off
    for h, acc in H.items():
      if t < h:
        acc["n"] += mf
        acc["sat"] += sat.float() * mf[:, None]
        acc["u"] += (u * mf[:, None]).sum(0).double()
        acc["err"] += (err * mf[:, None]).sum(0).double()
        acc["qd"] += (snap["qd"].abs() * mf[:, None]).sum(0).double()
        acc["lim"] += (atl.float() * mf[:, None]).sum(0).double()
        acc["hist"] += torch.bincount(bins[m].reshape(-1), minlength=16 * NBINS).double()
    first = dones.clone()
    qprev = q.clone()
    active &= ~dones
  cfg.func = orig  # break the probe's reference cycle so the environment's GPU memory can be freed
  snap.clear()
  env.close()
  out = {"yaw_inc": yaw.cpu().numpy(), "end_step": end_step.cpu().numpy(), "end_code": end_code.cpu().numpy(),
         "cache_id": cache, "dt": np.float32(raw.step_dt), "joint_names": np.array(jn),
         "blocked_idx_packed": np.packbits(blocked.cpu().numpy(), axis=0), "T": np.int64(T)}
  for h, acc in H.items():
    for kk, v in acc.items():
      out[f"h{h}_{kk}"] = v.cpu().numpy()
  return out


def summarise(tr: dict) -> dict:
  import math

  from in_hand_rotation_mjlab.grasp_rotation_v2.rollout import summary_per_env

  dt, T = float(tr["dt"]), int(tr["T"])
  jn = [str(x) for x in tr["joint_names"]]
  blocked = np.unpackbits(tr["blocked_idx_packed"], axis=0, count=T).astype(bool)  # (T, n, 2)
  jam = jam_per_env(blocked, tr["end_step"], dt)
  res = {}
  for h, tag in ((600, "30s"), (T, f"{T * dt:.0f}s")):
    s = summary_per_env(tr, h)
    nn = tr[f"h{h}_n"]
    surv = s["survival_s"] >= h * dt - 1e-6
    hist = tr[f"h{h}_hist"].reshape(16, NBINS)
    tot = hist.sum()
    allh = hist.sum(0)
    cdf = np.cumsum(allh) / max(allh.sum(), 1)
    pct = lambda c, p: float((np.searchsorted(c, p / 100.0) + 0.5) / NBINS)  # noqa: E731
    satj = tr[f"h{h}_sat"].sum(0) / max(nn.sum(), 1)
    r = {k: float(v.mean()) for k, v in s.items()}
    r.update({"p_held": float(1 - s["failure"].mean()), "completed": float(surv.mean()),
              "turns_if_survived": float(s["turns"][surv].mean()) if surv.any() else math.nan,
              "util_mean": float(tr[f"h{h}_u"].sum() / max(16 * nn.sum(), 1)),
              "util_p95": pct(cdf, 95), "util_p99": pct(cdf, 99),
              "util_mean_per_joint": dict(zip(jn, (tr[f"h{h}_u"] / max(nn.sum(), 1)).tolist())),
              "util_p99_per_joint": {j: pct(np.cumsum(hist[i]) / max(hist[i].sum(), 1), 99) for i, j in enumerate(jn)},
              "sat_frac_all": float(tr[f"h{h}_sat"].sum() / max(16 * nn.sum(), 1)),
              "sat_frac_per_joint": dict(zip(jn, satj.tolist())),
              "sat_if_mcp": float(satj[jn.index("if_mcp")]), "sat_if_pip": float(satj[jn.index("if_pip")]),
              "sat_th_axl": float(satj[jn.index("th_axl")]),
              "index_blocked_frac": float(blocked[:h].any(-1).sum() / max(nn.sum(), 1)),
              "jam_proxy": float(np.mean(jam[f"jam_{tag}"])),
              "tracking_err_mean": float(tr[f"h{h}_err"].sum() / max(16 * nn.sum(), 1)),
              "tracking_err_per_joint": dict(zip(jn, (tr[f"h{h}_err"] / max(nn.sum(), 1)).tolist())),
              "joint_speed_mean": float(tr[f"h{h}_qd"].sum() / max(16 * nn.sum(), 1)),
              "cmd_limit_occupancy": float(tr[f"h{h}_lim"].sum() / max(16 * nn.sum(), 1)),
              "alive_samples": float(nn.sum()), "hist_total": float(tot)})
    res[tag] = r
  return res


_AUDIT = None


def _audit():
  global _AUDIT
  if _AUDIT is None:
    import importlib.util
    p = Path(__file__).resolve().parents[5] / "LEAP_Hand_RL_Deploy_Kit" / "kit" / "audit_hardware_logs.py"
    spec = importlib.util.spec_from_file_location("audit_hardware_logs", p)
    import sys
    _AUDIT = importlib.util.module_from_spec(spec)
    sys.modules["audit_hardware_logs"] = _AUDIT  # dataclasses need the module registered
    spec.loader.exec_module(_AUDIT)
  return _AUDIT


def jam_per_env(blocked: np.ndarray, end_step: np.ndarray, dt: float) -> dict[str, np.ndarray]:
  """Per environment: any index-jam episode of >= 1 s (audit detector, index MCP/PIP columns only),
  within the first 30 s and over the whole rollout.  Samples after the first episode ended are excluded."""
  A = _audit()
  T, n, _ = blocked.shape
  hw = list(A.HW_JOINTS)
  cols = [hw.index("if_mcp"), hw.index("if_pip")]
  j30, jall, longest, onset = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n), np.full(n, np.nan)
  for e in range(n):
    L = T if end_step[e] < 0 else int(end_step[e]) + 1
    b = blocked[:L, e]
    if not b.any():
      continue
    full = np.zeros((L, 16), bool)
    full[:, cols] = b
    eps = A.jam_episodes(full, np.arange(L) * dt, dt)
    jall[e] = any(ep["duration_s"] >= 1.0 for ep in eps)
    onset[e] = min((ep["start_s"] for ep in eps if ep["duration_s"] >= 1.0), default=np.nan)
    longest[e] = max((ep["duration_s"] for ep in eps), default=0.0)
    L30 = min(L, int(round(30.0 / dt)))  # the detector on the first 30 s alone (as a 30-s log would be)
    eps30 = A.jam_episodes(full[:L30], np.arange(L30) * dt, dt) if full[:L30].any() else []
    j30[e] = any(ep["duration_s"] >= 1.0 for ep in eps30)
  return {"jam_30s": j30, "jam_60s": jall, "longest_jam_s": longest, "first_jam_onset_s": onset}


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--items")
  ap.add_argument("--make-items", action="store_true")
  ap.add_argument("--policies", default="A1,A2,A3,R1,C5_1,C5_2,C5_3")
  ap.add_argument("--out", default=str(E / "items" / "effort_v1_eval.json"))
  ap.add_argument("--device", default="cuda:0")
  a = ap.parse_args()
  if a.make_items:
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(make_items(a.policies.split(",")), indent=1))
    print("wrote", a.out)
    return
  items = json.loads(Path(a.items).read_text())
  (E / "results" / "items" / ".claims").mkdir(parents=True, exist_ok=True)
  (E / "data").mkdir(exist_ok=True)
  for it in items:
    key = it["key"]
    done = E / "results" / "items" / f"{key}.json"
    if done.exists():
      continue
    claim = E / "results" / "items" / ".claims" / key
    if claim.exists() and time.time() - claim.stat().st_mtime > 3 * 3600:
      claim.unlink(missing_ok=True)
    try:
      os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
      continue
    t0 = time.time()
    ck = policy_path(it["policy"])
    sha = hashlib.sha256(Path(ck).read_bytes()).hexdigest()
    tr = rollout(it["task"], ck, it["seed"], it["num_envs"], it["steps"], a.device, it["preset"],
                 assignment(it["assign"], it["num_envs"]))
    np.savez_compressed(E / "data" / f"{key}.npz", **tr)
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    summ = summarise(tr)
    done.write_text(json.dumps({"item": it, "checkpoint": ck, "sha256": sha, "summary": summ,
                                "host": os.uname().nodename, "seconds": round(time.time() - t0, 1)}, indent=1))
    s = summ["30s"]
    print(f"[{key}] 30s turns {s['turns']:.3f} fail {s['failure']:.3f} jam {s['jam_proxy']:.4f} "
          f"idx-sat {s['sat_if_mcp']:.4f}/{s['sat_if_pip']:.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
  main()
