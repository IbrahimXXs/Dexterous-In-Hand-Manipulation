#!/usr/bin/env python3
"""Phase 2: why does Retention-V1 (R1) rotate more slowly than A?  Uses the Phase 1
development rollouts (seeds 6101+6102, all 239 grasps equally, Mjlab-Leap-RetV1-A reward
terms logged with the frozen A1 curriculum weights).

1. reward composition per held step; retention penalty split into its position and tilt
   parts (recomputed with retention_v1.rewards.retention_excess and checked against the log);
2. activation rates of the parts, of the yaw-reward clip (4-step rate >= 0.25 rad/s) and of
   the drift/tilt gate (x0.1);
3. rotation speed in matched states (tilt x palm-position bins): R1's speed in A's state
   occupancy (direct standardisation) separates "different states" from "slower in the
   same states";
4. movement and contact timing (joint speed, |action|, saturation, contact switches);
5. reward-variant audit (H1-H4): 20-s episode returns of A, R1 and the hold policy under
   each variant, recomputed from the logged signals.

    python experiments/grasp_rotation_v2/scripts/analyze_reward.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

E = Path("experiments/grasp_rotation_v2")
POL = ("A1", "A2", "A3", "R1", "hold")
SEEDS = (6101, 6102)
DT = 0.05
W_ROT, CLIP, GATE_D, GATE_T = 1.25, 0.25, 0.02, 0.35
W_RET, W_GL = -0.05, -10.0


def load(pol, seed):
  return dict(np.load(E / "data" / f"p1dev_{pol}_s{seed}.npz"))


def parts(tr, **kw):
  from in_hand_rotation_mjlab.retention_v1.rewards import retention_excess

  pos = torch.from_numpy(tr["pos_palm"].astype(np.float32))
  tilt = torch.from_numpy(tr["tilt"].astype(np.float32))
  full = retention_excess(pos, tilt, **kw).numpy()
  pos_only = retention_excess(pos, torch.zeros_like(tilt), **kw).numpy()
  return pos_only, full - pos_only


def rate4(yaw):
  out = np.zeros_like(yaw)
  for k in range(4):
    out[k:] += yaw[: yaw.shape[0] - k]
  cnt = np.minimum(np.arange(1, yaw.shape[0] + 1), 4)[:, None]
  return out / cnt / DT


def rot_reward(tr, clip=CLIP, gate=True):
  r = np.clip(rate4(tr["yaw_inc"].astype(np.float64)), -clip, clip)
  if gate:
    inside = (tr["drift"].astype(np.float32) <= GATE_D) & (tr["tilt"].astype(np.float32) <= GATE_T)
    r = r * np.where(inside, 1.0, 0.1)
  return W_ROT * r * DT


def episode_returns(tr, per_step, horizon=400):
  """Return over the first 20 s (training episode) of the first episode; a failing
  environment stops accumulating at its failure step (inclusive)."""
  alive_or_end = tr["alive"][:horizon].astype(bool)
  e = tr["end_step"]
  endmask = np.zeros_like(alive_or_end)
  idx = np.where((e >= 0) & (e < horizon))[0]
  endmask[e[idx], idx] = True
  m = alive_or_end | endmask
  return (per_step[:horizon] * m).sum(0)


def main():
  names = None
  res = {"policies": {}, "variants": {}}
  bins_t = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8])
  occ, spd = {}, {}
  variant_ret = {}
  for pol in POL:
    acc = {k: [] for k in ("terms", "pos_part", "tilt_part", "held")}
    stats = {}
    tl = {}
    for seed in SEEDS:
      tr = load(pol, seed)
      names = [str(x) for x in tr["reward_names"]]
      held = tr["alive"].astype(bool)
      R = tr["reward"].astype(np.float32)
      pos_p, tilt_p = parts(tr)
      if seed == SEEDS[0]:
        # consistency of the recomputed retention penalty with the logged term
        logged = R[..., names.index("retention_penalty")]
        recomputed = W_RET * (pos_p + tilt_p) * DT
        m = held.copy()
        m[:3] = False
        stats["retention_recompute_max_abs_err"] = float(np.abs(logged - recomputed)[m].max())
        stats["retention_recompute_rel_err_mean"] = float(np.abs(logged - recomputed)[m].sum() / max(1e-9, np.abs(logged)[m].sum()))
        rr = rot_reward(tr)
        lr = R[..., names.index("rotate_finite_diff")]
        stats["rot_reward_recompute_mean_abs_err"] = float(np.abs(rr - lr)[m].mean())
        stats["rot_reward_logged_mean"] = float(lr[m].mean())
      r4 = rate4(tr["yaw_inc"].astype(np.float64))
      drift, tilt = tr["drift"].astype(np.float32), tr["tilt"].astype(np.float32)
      gate_out = (drift > GATE_D) | (tilt > GATE_T)
      z = (tr["pos_palm"].astype(np.float32) - np.array([0.0417, 0.0022, -0.0543])) / np.array([0.0099, 0.0099, 0.0032])
      md = np.sqrt((z ** 2).sum(-1))
      yaw_rate = tr["yaw_inc"] / DT
      for k, v in {"r4": r4, "yaw_rate": yaw_rate, "sat": (r4 >= CLIP).astype(float), "gate_out": gate_out.astype(float),
                   "pos_on": (pos_p > 0).astype(float), "tilt_on": (tilt_p > 0).astype(float), "tilt": tilt, "md": md,
                   "drift": drift, "joint_speed": tr["joint_speed"].astype(np.float32),
                   "action_abs": tr["action_abs"].astype(np.float32)}.items():
        tl.setdefault(k, []).append(v[held])
      bits = tr["contact"]
      nct = sum(((bits >> i) & 1) for i in range(4)).astype(np.float32)
      sw = np.zeros_like(nct)
      sw[1:] = sum((((bits[1:] >> i) & 1) != ((bits[:-1] >> i) & 1)) for i in range(4))
      tl.setdefault("n_contact", []).append(nct[held])
      tl.setdefault("contact_switch_per_s", []).append(sw[held] / DT)
      if "action" in tr:
        tl.setdefault("act_sat", []).append((np.abs(tr["action"].astype(np.float32)) > 0.99).mean(-1)[held])
      # matched-state bins: tilt bin x md bin (md < 1.5, 1.5-3, >= 3)
      tb = np.clip(np.digitize(tilt, bins_t) - 1, 0, len(bins_t) - 2)
      mb = np.digitize(md, [1.5, 3.0])
      cell = tb * 3 + mb
      for c in range(len(bins_t) * 3):
        k = held & (cell == c)
        occ.setdefault(pol, np.zeros(len(bins_t) * 3))[c] += k.sum()
        spd.setdefault(pol, np.zeros(len(bins_t) * 3))[c] += yaw_rate[k].sum()
      acc["terms"].append(R[held].mean(0) * held.sum())
      acc["held"].append(held.sum())
      acc["pos_part"].append((W_RET * pos_p * DT)[held].sum())
      acc["tilt_part"].append((W_RET * tilt_p * DT)[held].sum())
      # variant returns (20-s episodes), recomputed from logged signals
      base_other = sum(R[..., names.index(n)] for n in names if n not in ("rotate_finite_diff", "retention_penalty", "grasp_loss"))
      gl = R[..., names.index("grasp_loss")]
      rot = R[..., names.index("rotate_finite_diff")]
      variants = {
        "RetV1 (as trained)": rot + base_other + W_RET * (pos_p + tilt_p) * DT + gl,
        "A reward (no retention terms)": rot + base_other,
        "H1: no position part": rot + base_other + W_RET * tilt_p * DT + gl,
        "H1b: position part x0.25": rot + base_other + W_RET * (0.25 * pos_p + tilt_p) * DT + gl,
        "H2: tilt margin 0.6 rad": rot + base_other + W_RET * (pos_p + parts(tr, tilt_margin=0.6)[1]) * DT + gl,
        "H3: grasp loss only": rot + base_other + gl,
        "H4: yaw clip 0.5 (RetV1 otherwise)": rot_reward(tr, clip=0.5) + base_other + W_RET * (pos_p + tilt_p) * DT + gl,
      }
      for vn, ps in variants.items():
        variant_ret.setdefault(vn, {}).setdefault(pol, []).append(episode_returns(tr, ps))
    H = float(np.sum(acc["held"]))
    terms = np.sum(acc["terms"], 0) / H
    d = {"held_steps": int(H), "per_held_step_by_term": dict(zip(names, terms.round(6).tolist())),
         "retention_position_part": float(np.sum(acc["pos_part"]) / H), "retention_tilt_part": float(np.sum(acc["tilt_part"]) / H)}
    rot_mean = d["per_held_step_by_term"]["rotate_finite_diff"]
    d["retention_share_of_rotation_reward"] = float(-(d["retention_position_part"] + d["retention_tilt_part"]) / max(1e-9, rot_mean))
    for k, v in tl.items():
      x = np.concatenate(v).astype(np.float64)
      d[f"mean_{k}"] = float(x.mean())
    x = np.concatenate(tl["yaw_rate"])
    d["yaw_rate_quantiles_held"] = np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9]).round(3).tolist()
    r4 = np.concatenate(tl["r4"])
    d["r4_quantiles_held"] = np.quantile(r4, [0.1, 0.25, 0.5, 0.75, 0.9]).round(3).tolist()
    d.update(stats)
    res["policies"][pol] = d
  # matched-state standardisation
  A_occ = sum(occ[p] for p in ("A1", "A2", "A3"))
  A_spd = sum(spd[p] for p in ("A1", "A2", "A3")) / np.maximum(A_occ, 1)
  R_spd = spd["R1"] / np.maximum(occ["R1"], 1)
  ok = (A_occ > 200) & (occ["R1"] > 200)
  wA = A_occ * ok / (A_occ * ok).sum()
  wR = occ["R1"] * ok / (occ["R1"] * ok).sum()
  res["matched_states"] = {
    "A_speed_in_A_states": float((A_spd * wA).sum()), "R1_speed_in_R1_states": float((R_spd * wR).sum()),
    "R1_speed_in_A_states": float((R_spd * wA).sum()), "A_speed_in_R1_states": float((A_spd * wR).sum()),
    "cells_used": int(ok.sum()), "share_of_held_steps_covered_A": float((A_occ * ok).sum() / A_occ.sum()),
    "occupancy_A": (A_occ / A_occ.sum()).round(4).tolist(), "occupancy_R1": (occ["R1"] / occ["R1"].sum()).round(4).tolist(),
    "speed_A": A_spd.round(4).tolist(), "speed_R1": R_spd.round(4).tolist(), "tilt_bins": bins_t.tolist(), "md_bins": [1.5, 3.0]}
  for vn, d in variant_ret.items():
    m = {p: float(np.concatenate(v).mean()) for p, v in d.items()}
    m["A_mean"] = float(np.mean([m[p] for p in ("A1", "A2", "A3")]))
    m["A_minus_R1"] = m["A_mean"] - m["R1"]
    m["R1_minus_hold"] = m["R1"] - m["hold"]
    res["variants"][vn] = m
  (E / "results" / "reward_analysis.json").write_text(json.dumps(res, indent=1))
  for p, d in res["policies"].items():
    print(p, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items() if k.startswith(("mean_", "retention_", "rot_", "yaw_rate_q", "r4_q"))})
  print(json.dumps({k: v for k, v in res["matched_states"].items() if not isinstance(v, list)}, indent=1))
  for vn, m in res["variants"].items():
    print(f"{vn:38s} " + " ".join(f"{k} {v:+.3f}" for k, v in m.items()))


if __name__ == "__main__":
  main()
