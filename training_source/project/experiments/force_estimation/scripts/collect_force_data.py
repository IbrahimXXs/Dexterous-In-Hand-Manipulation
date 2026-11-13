#!/usr/bin/env python3
"""Collect (320-D actor observation, fingertip normal forces) pairs from trained study policies.

Environment: the completed study's ``Mjlab-Leap-Scratch-<A|C|D>`` env (training DR,
observation noise and delays, 20 s episodes, curriculum at final weights, optional
robustness preset) + a passive contact-force sensor (labels only).  Policies are the
existing, unmodified study checkpoints (read only).

Per row (control step t, env i):
  obs.npy             [T, N, 320] float32  estimator INPUT (actor observation)
  force_mean          [T, N, 4]  mean normal force over the 10 physics substeps of the last
                                 control interval (PRIMARY LABEL), N, order thumb/index/middle/ring
  force_now           [T, N, 4]  normal force at the observation instant (post-step forward)
  force_max           [T, N, 4]  max over the substeps
  contact_frac        [T, N, 4]  fraction of substeps with non-zero normal force
  found_now           [T, N, 4]  number of fingertip-cube contacts (before top-k reduction)
  tangential_now      [T, N, 4]  summed friction-force magnitude
  other_normal_now    [T, N]     normal force between cube and non-fingertip hand links (palm ...)
  cube_lin_vel/ang_vel/pos (palm frame), step_in_episode, episode_id, end_reason, dr_*  - metadata
Nothing except obs.npy may be used as estimator input.

    python collect_force_data.py --condition A --checkpoint <pt> --split train --seed 21000 \
        --num-envs 256 --num-steps 600 --out experiments/force_estimation/data
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from in_hand_rotation_mjlab.estimation.layout import ACTOR_OBS_DIM, JOINT_NAMES
from in_hand_rotation_mjlab.estimation.sim import GroundTruth
from in_hand_rotation_mjlab.force_estimation.constants import FORCE_FINGER_NAMES, FORCE_SENSOR_SLOTS
from in_hand_rotation_mjlab.force_estimation.labels import (
  ForceLabeler,
  make_force_env,
  raw_contact_normal_forces,
  raw_other_hand_cube_normal_force,
)
from in_hand_rotation_mjlab.scratch_study.evaluation import load_study_policy


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--condition", required=True, choices=["A", "C", "D"])
  p.add_argument("--checkpoint", required=True)
  p.add_argument("--policy-name", required=True, help="e.g. A_seed2_it600 (recorded in meta)")
  p.add_argument("--split", required=True)
  p.add_argument("--seed", type=int, required=True)
  p.add_argument("--preset", default="nominal")
  p.add_argument("--num-envs", type=int, default=256)
  p.add_argument("--num-steps", type=int, default=600)
  p.add_argument("--noise-scale", type=float, default=1.0)
  p.add_argument("--stochastic-fraction", type=float, default=0.5)
  p.add_argument("--out", default="experiments/force_estimation/data")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--audit-every", type=int, default=25, help="cross-check sensor vs raw contact list")
  a = p.parse_args()

  out = Path(a.out) / a.split / f"{a.policy_name}__{a.preset}__seed{a.seed}"
  if (out / "meta.json").exists():
    print(f"[collect] exists, skipping: {out}")
    return
  out.mkdir(parents=True, exist_ok=True)
  torch.manual_seed(a.seed)
  env, raw, agent_cfg = make_force_env(a.condition, a.num_envs, a.seed, a.device, preset=a.preset)
  if tuple(raw.scene["robot"].joint_names) != JOINT_NAMES:
    raise RuntimeError("unexpected joint order")
  lab = ForceLabeler(raw)
  gt = GroundTruth(raw)
  actor = load_study_policy(env, agent_cfg, a.checkpoint, a.device)
  n, T, dev = a.num_envs, a.num_steps, a.device
  std = actor.std.detach() * a.noise_scale
  stochastic = torch.rand(n, device=dev) < a.stochastic_fraction

  obs_mm = np.lib.format.open_memmap(out / "obs.npy", mode="w+", dtype=np.float32, shape=(T, n, ACTOR_OBS_DIM))
  z = lambda *s, dt=torch.float32: torch.zeros(T, n, *s, device=dev, dtype=dt)  # noqa: E731
  buf = {k: z(4) for k in ("force_mean", "force_now", "force_max", "contact_frac", "tangential_now")}
  buf["found_now"] = z(4, dt=torch.int16)
  buf["other_normal_now"] = z()
  for k in ("cube_lin_vel", "cube_ang_vel", "cube_pos"):
    buf[k] = z(3)
  step_in_ep = z(dt=torch.int32)
  episode_id = z(dt=torch.int64)
  end_reason = z(dt=torch.int8)
  dr: dict[str, torch.Tensor] = {}
  env_ids = torch.arange(n, device=dev, dtype=torch.int64)
  ep_counter = torch.zeros(n, device=dev, dtype=torch.int64)
  raw.episode_length_buf = torch.randint_like(raw.episode_length_buf, high=int(raw.max_episode_length))
  obs = env.get_observations()
  audit_max_diff, neg_total, overflow_rows = 0.0, 0, 0
  chunk, chunk_start, flush = [], 0, 50
  t0 = time.time()
  with torch.inference_mode():
    for t in range(T):
      mean = actor(obs)
      action = torch.where(stochastic[:, None], mean + torch.randn_like(mean) * std, mean)
      obs, _, dones, _ = env.step(action)
      dones = dones.bool()
      if t > 0:
        end_reason[t - 1] = gt.end_reasons(dones)
      ep_counter += dones.long()
      now = lab.now()
      itv = lab.interval()
      buf["force_now"][t] = now["normal"]
      buf["tangential_now"][t] = now["tangential"]
      buf["found_now"][t] = now["found"].to(torch.int16)
      buf["force_mean"][t] = itv["mean"]
      buf["force_max"][t] = itv["max"]
      buf["contact_frac"][t] = itv["contact_frac"]
      buf["other_normal_now"][t] = raw_other_hand_cube_normal_force(raw)
      neg_total += int(now["neg"].sum())
      overflow_rows += int((now["found"] > FORCE_SENSOR_SLOTS).any(-1).sum())
      if t % a.audit_every == 0:
        audit_max_diff = max(audit_max_diff, float((raw_contact_normal_forces(raw) - now["normal"]).abs().max()))
      g = gt.labels(obs["critic"])
      buf["cube_lin_vel"][t] = g["lin_vel"]
      buf["cube_ang_vel"][t] = g["ang_vel"]
      buf["cube_pos"][t] = g["pos"]
      step_in_ep[t] = raw.episode_length_buf.to(torch.int32)
      episode_id[t] = env_ids * 100_000 + ep_counter
      for k, v in gt.randomization().items():
        if k not in dr:
          dr[k] = z()
        dr[k][t] = v
      chunk.append(obs["actor"].clone())
      if len(chunk) == flush or t == T - 1:
        obs_mm[chunk_start : t + 1] = torch.stack(chunk).cpu().numpy()
        chunk_start, chunk = t + 1, []
      if (t + 1) % 100 == 0:
        print(f"[collect] {t + 1}/{T}  {n * (t + 1) / (time.time() - t0):,.0f} rows/s", flush=True)
  obs_mm.flush()
  del obs_mm
  for k, v in buf.items():
    np.save(out / f"{k}.npy", v.cpu().numpy())
  np.save(out / "step_in_episode.npy", step_in_ep.cpu().numpy())
  np.save(out / "episode_id.npy", episode_id.cpu().numpy())
  np.save(out / "end_reason.npy", end_reason.cpu().numpy())
  for k, v in dr.items():
    np.save(out / f"{k}.npy", v.cpu().numpy())
  np.save(out / "stochastic_mask.npy", stochastic.cpu().numpy())
  meta = {
    "condition": a.condition, "checkpoint": str(Path(a.checkpoint).resolve()), "policy_name": a.policy_name,
    "split": a.split, "seed": a.seed, "preset": a.preset, "num_envs": n, "num_steps": T,
    "noise_scale": a.noise_scale, "stochastic_fraction": a.stochastic_fraction,
    "control_dt": float(raw.step_dt), "physics_dt": float(raw.physics_dt), "decimation": int(raw.cfg.decimation),
    "finger_order": list(FORCE_FINGER_NAMES), "primary_label": "force_mean",
    "sensor_slots": FORCE_SENSOR_SLOTS, "rows_with_slot_overflow": overflow_rows,
    "negative_normal_values_clipped": neg_total,
    "audit_sensor_vs_raw_max_abs_diff_N": audit_max_diff,
    "actor_terms": gt.actor_terms, "actor_term_dims": gt.actor_term_dims, "joint_names": list(JOINT_NAMES),
    "dr_fields": sorted(dr), "elapsed_s": time.time() - t0, "created": datetime.now().isoformat(timespec="seconds"),
    "host_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
  }
  (out / "meta.json").write_text(json.dumps(meta, indent=2))
  print(f"[collect] wrote {out} ({T * n:,} rows) audit max|sensor-raw|={audit_max_diff:.2e} N "
        f"overflow rows={overflow_rows} neg clipped={neg_total}")


if __name__ == "__main__":
  main()
