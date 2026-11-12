#!/usr/bin/env python3
"""Collect (actor observation, privileged cube state) pairs with model_400.pt.

Runs the *original* task (training configuration: domain randomisation,
observation noise and delays, 20 s episodes, grasp-cache resets) in parallel
CUDA environments and records, at every control step, the 320-D actor
observation together with ground truth read from the same physics state.

One invocation = one seed = one split shard.  Use different seeds for train /
val / test so that no episode, environment instance or startup-randomisation
sample is shared between splits (see ``collect_all.sh``).

Example::

    CUDA_VISIBLE_DEVICES=1 python scripts/estimation/collect_estimator_data.py \
        --split train --seed 1001 --num-envs 2048 --num-steps 800 \
        --output-root data/estimator
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

from in_hand_rotation_mjlab.estimation.layout import (
  ACTOR_OBS_DIM,
  FINGERTIP_NAMES,
  JOINT_NAMES,
)
from in_hand_rotation_mjlab.estimation.sim import (
  TASK_ID,
  GroundTruth,
  load_policy,
  make_env,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CKPT = ROOT / "checkpoint" / "model_400.pt"


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--split", required=True, choices=["train", "val", "test"])
  p.add_argument("--seed", type=int, required=True)
  p.add_argument("--num-envs", type=int, default=2048)
  p.add_argument("--num-steps", type=int, default=800, help="control steps (20 Hz)")
  p.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
  p.add_argument("--output-root", default="data/estimator")
  p.add_argument("--device", default="cuda:0")
  p.add_argument(
    "--stochastic-fraction",
    type=float,
    default=0.5,
    help="fraction of envs that sample actions from the policy's Gaussian "
    "(covers the PPO fine-tuning state distribution); the rest act on the mean",
  )
  p.add_argument("--noise-scale", type=float, default=1.0, help="multiplier on the checkpoint std")
  p.add_argument("--flush-every", type=int, default=50, help="steps buffered on GPU before disk write")
  p.add_argument(
    "--no-random-episode-start",
    action="store_true",
    help="disable random initial episode lengths (training uses them)",
  )
  p.add_argument(
    "--original-reset-anchor",
    action="store_true",
    help="keep the original cube_pose_deviation reset anchor (stale pose; see "
    "estimation/env_fixes.py).  Default: corrected anchor.",
  )
  p.add_argument(
    "--task",
    default=TASK_ID,
    help="environment; Mjlab-Leap-Scratch-<A|C|D> uses the corrected from-scratch study env and loader",
  )
  p.add_argument("--overwrite", action="store_true")
  return p.parse_args()


def main() -> None:
  args = parse_args()
  out_dir = Path(args.output_root) / args.split / f"seed_{args.seed}"
  if (out_dir / "meta.json").exists() and not args.overwrite:
    raise FileExistsError(f"{out_dir} already holds a finished run (use --overwrite)")
  out_dir.mkdir(parents=True, exist_ok=True)
  torch.manual_seed(args.seed)

  scratch = args.task.startswith("Mjlab-Leap-Scratch-")
  if scratch:
    from in_hand_rotation_mjlab.scratch_study.evaluation import load_study_policy, make_study_env

    env, raw, agent_cfg = make_study_env(args.task.rsplit("-", 1)[-1], args.num_envs, args.seed, args.device)
  else:
    env, raw, agent_cfg = make_env(
      args.num_envs,
      args.seed,
      args.device,
      fix_reset_anchor=not args.original_reset_anchor,
    )
  gt = GroundTruth(raw)
  if tuple(n.replace("_tip", "") for n in gt.contact_names) != FINGERTIP_NAMES:
    raise RuntimeError(f"Unexpected contact slot order {gt.contact_names}")
  robot_joints = tuple(raw.scene["robot"].joint_names)
  if robot_joints != JOINT_NAMES:
    raise RuntimeError(f"Unexpected joint order {robot_joints}")
  if scratch:
    actor = load_study_policy(env, agent_cfg, args.checkpoint, args.device)
  else:
    _, actor = load_policy(env, agent_cfg, args.checkpoint, args.device)

  n, T, dev = args.num_envs, args.num_steps, args.device
  std = actor.std.detach() * args.noise_scale
  stochastic = torch.rand(n, device=dev) < args.stochastic_fraction

  # Large array streamed to disk; everything else kept on the GPU.
  obs_mm = np.lib.format.open_memmap(
    out_dir / "obs.npy", mode="w+", dtype=np.float32, shape=(T, n, ACTOR_OBS_DIM)
  )
  zf = lambda *s: torch.zeros(T, n, *s, device=dev, dtype=torch.float32)  # noqa: E731
  buf = {
    "pos": zf(3),
    "quat": zf(4),
    "lin_vel": zf(3),
    "ang_vel": zf(3),
    "cube_pos_w": zf(3),
    "cube_quat_w": zf(4),
  }
  contact = torch.zeros(T, n, 4, device=dev, dtype=torch.uint8)
  contact_valid = torch.zeros(T, n, device=dev, dtype=torch.uint8)
  episode_id = torch.zeros(T, n, device=dev, dtype=torch.int64)
  step_in_ep = torch.zeros(T, n, device=dev, dtype=torch.int32)
  end_reason = torch.zeros(T, n, device=dev, dtype=torch.int8)
  dr: dict[str, torch.Tensor] = {}
  obs_chunk = torch.zeros(args.flush_every, n, ACTOR_OBS_DIM, device=dev)

  env_ids = torch.arange(n, device=dev, dtype=torch.int64)
  ep_counter = torch.zeros(n, device=dev, dtype=torch.int64)
  if not args.no_random_episode_start:
    raw.episode_length_buf = torch.randint_like(
      raw.episode_length_buf, high=int(raw.max_episode_length)
    )
  obs = env.get_observations()

  t0 = time.time()
  chunk_start = 0
  with torch.inference_mode():
    for t in range(T):
      mean = actor(obs)
      noise = torch.randn_like(mean) * std
      action = torch.where(stochastic[:, None], mean + noise, mean)
      obs, _, dones, _ = env.step(action)
      dones = dones.bool()
      reasons = gt.end_reasons(dones)
      if t > 0:
        end_reason[t - 1] = reasons  # previous row was the last of its episode
      ep_counter += dones.long()

      labels = gt.labels(obs["critic"])
      for k in ("pos", "quat", "lin_vel", "ang_vel"):
        buf[k][t] = labels[k]
      contact[t] = labels["contact"]
      pw, qw = gt.world_pose()
      buf["cube_pos_w"][t] = pw
      buf["cube_quat_w"][t] = qw
      sie = raw.episode_length_buf.to(torch.int32)
      step_in_ep[t] = sie
      contact_valid[t] = (sie > 0).to(torch.uint8)
      episode_id[t] = env_ids * 100_000 + ep_counter
      for k, v in gt.randomization().items():
        if k not in dr:
          dr[k] = torch.zeros(T, n, device=dev)
        dr[k][t] = v
      obs_chunk[t - chunk_start] = obs["actor"]

      if t - chunk_start + 1 == args.flush_every or t == T - 1:
        obs_mm[chunk_start : t + 1] = obs_chunk[: t - chunk_start + 1].cpu().numpy()
        chunk_start = t + 1
      if (t + 1) % 100 == 0 or t == T - 1:
        el = time.time() - t0
        print(
          f"[collect] step {t + 1}/{T}  {n * (t + 1) / el:,.0f} samples/s  "
          f"episodes ended {int((end_reason[:t] > 0).sum())}",
          flush=True,
        )

  obs_mm.flush()
  del obs_mm
  save = lambda name, x: np.save(out_dir / f"{name}.npy", x.cpu().numpy())  # noqa: E731
  for k, v in buf.items():
    save(k, v)
  save("contact", contact)
  save("contact_valid", contact_valid)
  save("episode_id", episode_id)
  save("step_in_episode", step_in_ep)
  save("end_reason", end_reason)
  for k, v in dr.items():
    save(k, v)
  np.save(out_dir / "stochastic_mask.npy", stochastic.cpu().numpy())

  er = end_reason.cpu().numpy()
  counts = {int(c): int((er == c).sum()) for c in np.unique(er) if c != 0}
  meta = {
    "task": args.task,
    "checkpoint": str(Path(args.checkpoint).resolve()),
    "split": args.split,
    "seed": args.seed,
    "num_envs": n,
    "num_steps": T,
    "control_dt": float(raw.step_dt),
    "env_config": "training (play=False), curriculum frozen, DR unchanged",
    "reset_anchor_fix": not args.original_reset_anchor,
    "stochastic_fraction": args.stochastic_fraction,
    "noise_scale": args.noise_scale,
    "random_episode_start": not args.no_random_episode_start,
    "actor_terms": gt.actor_terms,
    "actor_term_dims": gt.actor_term_dims,
    "critic_slices": {k: [v.start, v.stop] for k, v in gt.critic_slices.items()},
    "contact_slots": gt.contact_names,
    "joint_names": list(JOINT_NAMES),
    "end_reason_counts": counts,
    "dr_fields": sorted(dr),
    "elapsed_s": time.time() - t0,
    "created": datetime.now().isoformat(timespec="seconds"),
    "host_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
  }
  (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
  print(f"[collect] wrote {out_dir}  ({T * n:,} rows)  end reasons {counts}")


if __name__ == "__main__":
  main()
