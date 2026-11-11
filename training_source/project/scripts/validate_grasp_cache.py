#!/usr/bin/env python3
"""Validate cached in-hand reset states under a fixed command hold."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

import mjlab.tasks  # noqa: F401
import in_hand_rotation_mjlab.tasks  # noqa: F401


def _cube_hand_penetration(
  env: ManagerBasedRlEnv,
  cube_geom_id: int,
  robot_geom_ids: torch.Tensor,
) -> torch.Tensor:
  """Return deepest cube-hand penetration in each world, in metres."""
  result = torch.zeros(env.num_envs, device=env.device)
  nacon = int(env.sim.data.nacon[0])
  if nacon == 0:
    return result
  geom = env.sim.data.contact.geom[:nacon].to(torch.long)
  dist = env.sim.data.contact.dist[:nacon]
  world = env.sim.data.contact.worldid[:nacon].to(torch.long)
  first_cube = geom[:, 0] == cube_geom_id
  second_cube = geom[:, 1] == cube_geom_id
  other = torch.where(first_cube, geom[:, 1], geom[:, 0])
  relevant = (first_cube | second_cube) & torch.isin(other, robot_geom_ids)
  if torch.any(relevant):
    penetration = torch.clamp(-dist[relevant], min=0.0)
    result.scatter_reduce_(
      0, world[relevant], penetration, reduce="amax", include_self=True
    )
  return result


def _finite_per_env(*values: torch.Tensor) -> torch.Tensor:
  finite = torch.ones(values[0].shape[0], device=values[0].device, dtype=torch.bool)
  for value in values:
    finite &= torch.isfinite(value).reshape(value.shape[0], -1).all(dim=-1)
  return finite


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--task", default="Mjlab-Leap-Right-HandCube-Playground-Reorient"
  )
  parser.add_argument("--cache-file", required=True)
  parser.add_argument("--num-envs", type=int, default=256)
  parser.add_argument("--duration-s", type=float, default=10.0)
  parser.add_argument("--minimum-height", type=float, default=0.20)
  parser.add_argument("--maximum-palm-distance", type=float, default=0.15)
  parser.add_argument("--minimum-contacts", type=int, default=2)
  parser.add_argument("--penetration-limit-mm", type=float, default=2.0)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--seed", type=int, default=60)
  parser.add_argument("--json-output", default=None)
  args = parser.parse_args()

  cache_file = Path(args.cache_file).expanduser().resolve()
  if not cache_file.is_file():
    raise FileNotFoundError(cache_file)
  if args.num_envs < 1 or args.duration_s <= 0.0:
    raise ValueError("num-envs and duration-s must be positive")

  with np.load(cache_file, allow_pickle=False) as cache:
    required = {
      "cube_size", "joint_pos", "joint_pos_target", "cube_pose_rel",
      "joint_names", "fingertip_contact", "source_checkpoint",
    }
    missing = required.difference(cache.files)
    if missing:
      raise ValueError(f"cache is missing required arrays: {sorted(missing)}")
    cache_size = np.asarray(cache["cube_size"], dtype=np.float64)
    cache_contacts = np.asarray(cache["fingertip_contact"], dtype=np.float64)
    cache_summary = {
      "samples": int(cache_size.shape[0]),
      "cube_half_size_min_m": float(cache_size.min()),
      "cube_half_size_max_m": float(cache_size.max()),
      "per_fingertip_contact_fraction": cache_contacts.mean(axis=0).tolist(),
      "minimum_simultaneous_contacts": int(cache_contacts.sum(axis=1).min()),
      "mean_simultaneous_contacts": float(cache_contacts.sum(axis=1).mean()),
      "source_checkpoint": str(np.asarray(cache["source_checkpoint"]).item()),
    }

  configure_torch_backends()
  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  cache_reset = env_cfg.events["reset_from_grasp_cache"].params
  cache_reset["cache_file"] = str(cache_file)
  step_dt = env_cfg.sim.mujoco.timestep * env_cfg.decimation
  num_steps = math.ceil(args.duration_s / step_dt)
  env_cfg.episode_length_s = max(env_cfg.episode_length_s, args.duration_s + step_dt)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  robot = env.scene["robot"]
  cube = env.scene["cube"]
  sensor = env.scene["fingertip_cube_contact"]
  palm_geom_ids, _ = robot.find_geoms("palm_collision_.*", preserve_order=True)
  cube_geom_ids, _ = cube.find_geoms("cube_geom", preserve_order=True)
  if len(palm_geom_ids) == 0 or len(cube_geom_ids) != 1:
    raise RuntimeError("could not resolve palm or cube collision geometry")
  cube_geom_id = int(cube.indexing.geom_ids[cube_geom_ids[0]].item())
  robot_geom_ids = robot.indexing.geom_ids.flatten().to(env.device, dtype=torch.long)

  env.reset()
  action = torch.zeros((args.num_envs, env.action_manager.total_action_dim), device=env.device)
  active = torch.ones(args.num_envs, device=env.device, dtype=torch.bool)
  ever_nan = torch.zeros_like(active)
  ever_drop = torch.zeros_like(active)
  immediate_drop = torch.zeros_like(active)
  ever_contact = torch.zeros((args.num_envs, 4), device=env.device, dtype=torch.bool)
  contact_active_sum = torch.zeros(4, device=env.device)
  active_sample_count = torch.tensor(0.0, device=env.device)

  initial_found = sensor.data.found
  assert initial_found is not None
  initial_contacts = initial_found > 0
  initial_contact_count = initial_contacts.sum(dim=-1)
  max_penetration = _cube_hand_penetration(env, cube_geom_id, robot_geom_ids)
  initial_penetration = max_penetration.clone()

  with torch.inference_mode():
    for step in range(num_steps):
      _, _, terminated, time_out, _ = env.step(action)
      done = terminated.bool() | time_out.bool()
      palm_center = robot.data.geom_pos_w[:, palm_geom_ids].mean(dim=1)
      palm_distance = torch.linalg.vector_norm(
        cube.data.root_link_pos_w - palm_center, dim=-1
      )
      finite = _finite_per_env(
        cube.data.root_link_pos_w,
        cube.data.root_link_quat_w,
        cube.data.root_link_lin_vel_w,
        cube.data.root_link_ang_vel_w,
        robot.data.joint_pos,
        robot.data.joint_vel,
      )
      dropped_now = (
        done
        | (cube.data.root_link_pos_w[:, 2] < args.minimum_height)
        | (palm_distance > args.maximum_palm_distance)
      )
      nan_now = ~finite
      if step == 0:
        immediate_drop |= dropped_now | nan_now

      found = sensor.data.found
      assert found is not None
      contacts = found > 0
      valid_sample = active & ~dropped_now & ~nan_now
      ever_contact |= contacts & valid_sample[:, None]
      contact_active_sum += (contacts & valid_sample[:, None]).float().sum(dim=0)
      active_sample_count += valid_sample.float().sum()

      penetration = _cube_hand_penetration(env, cube_geom_id, robot_geom_ids)
      max_penetration = torch.maximum(max_penetration, penetration)
      ever_drop |= active & dropped_now
      ever_nan |= active & nan_now
      active &= ~(dropped_now | nan_now)

  limit_m = args.penetration_limit_mm / 1000.0
  result = {
    "task": args.task,
    "cache_file": str(cache_file),
    "cache": cache_summary,
    "num_envs": args.num_envs,
    "duration_s": num_steps * step_dt,
    "num_steps": num_steps,
    "step_dt_s": step_dt,
    "immediate_drop_count": int(immediate_drop.sum()),
    "immediate_drop_rate": float(immediate_drop.float().mean()),
    "retained_10s_count": int(active.sum()),
    "retention_rate": float(active.float().mean()),
    "ever_drop_count": int(ever_drop.sum()),
    "nan_environment_count": int(ever_nan.sum()),
    "initial_minimum_contact_fraction": float(
      (initial_contact_count >= args.minimum_contacts).float().mean()
    ),
    "initial_per_fingertip_contact_fraction": [
      float(x) for x in initial_contacts.float().mean(dim=0)
    ],
    "hold_per_fingertip_contact_fraction": [
      float(x) for x in (contact_active_sum / active_sample_count.clamp_min(1.0))
    ],
    "ever_per_fingertip_contact_fraction": [
      float(x) for x in ever_contact.float().mean(dim=0)
    ],
    "initial_max_cube_hand_penetration_mm": float(initial_penetration.max() * 1000.0),
    "max_cube_hand_penetration_mm": float(max_penetration.max() * 1000.0),
    "penetration_over_limit_count": int((max_penetration > limit_m).sum()),
    "penetration_limit_mm": args.penetration_limit_mm,
  }
  print(json.dumps(result, indent=2, sort_keys=True))
  if args.json_output:
    output = Path(args.json_output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"saved: {output}")
  env.close()


if __name__ == "__main__":
  main()
