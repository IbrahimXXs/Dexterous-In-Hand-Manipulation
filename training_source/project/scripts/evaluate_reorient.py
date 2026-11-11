#!/usr/bin/env python3
"""Evaluate an arbitrary SO(3) cube-reorientation checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_mul
from mjlab.utils.torch import configure_torch_backends

from in_hand_rotation_mjlab.tasks.hand_cube.mdp.pose_goal import (
  quat_angular_distance,
)

import mjlab.tasks  # noqa: F401
import in_hand_rotation_mjlab.tasks  # noqa: F401


def _prepare_agent_cfg(agent_cfg) -> dict:
  cfg = asdict(agent_cfg)
  for key in ("actor", "critic"):
    if cfg[key].get("class_name", "MLPModel") != "CNNModel":
      cfg[key].pop("cnn_cfg", None)
  return cfg


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint-file", required=True)
  parser.add_argument("--task", default="Mjlab-Leap-Right-HandCube-Reorient")
  parser.add_argument("--num-envs", type=int, default=600)
  parser.add_argument("--num-steps", type=int, default=400)
  parser.add_argument("--success-angle-deg", type=float, default=10.0)
  parser.add_argument("--success-dwell-steps", type=int, default=5)
  parser.add_argument("--min-target-angle-deg", type=float, default=10.0)
  parser.add_argument("--max-target-angle-deg", type=float, default=180.0)
  parser.add_argument("--max-initial-palm-distance", type=float, default=0.15)
  parser.add_argument("--min-initial-contacts", type=int, default=2)
  parser.add_argument("--max-reset-attempts", type=int, default=100)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--json-output", default=None)
  args = parser.parse_args()

  checkpoint = Path(args.checkpoint_file).expanduser().resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
  if args.num_envs <= 0 or args.num_steps <= 0:
    raise ValueError("num-envs and num-steps must be positive")
  if not 0.0 <= args.min_target_angle_deg <= args.max_target_angle_deg <= 180.0:
    raise ValueError("Target-angle range must lie in [0, 180] degrees.")

  configure_torch_backends()
  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  agent_cfg = load_rl_cfg(args.task)

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or OnPolicyRunner
  runner = runner_cls(env, _prepare_agent_cfg(agent_cfg), device=args.device)
  runner.load(str(checkpoint), map_location=args.device)
  policy = runner.get_inference_policy(device=args.device)

  sensor = raw_env.scene["fingertip_cube_contact"]
  cube = raw_env.scene["cube"]
  all_env_ids = torch.arange(args.num_envs, device=args.device)
  pending = torch.ones(args.num_envs, device=args.device, dtype=torch.bool)
  reset_attempts = torch.zeros(
    args.num_envs, device=args.device, dtype=torch.long
  )
  actor_terms = raw_env.observation_manager.active_terms["actor"]
  camera_term_name = (
    "camera_cube_pose"
    if "camera_cube_pose" in actor_terms
    else "camera_cube_position_error"
  )
  camera_pose_cfg = raw_env.observation_manager.get_term_cfg(
    "actor", camera_term_name
  )
  for _ in range(args.max_reset_attempts):
    pose_or_error = camera_pose_cfg.func(
      raw_env, **camera_pose_cfg.params
    )
    position = pose_or_error[:, :3]
    distance_ok = (
      torch.linalg.vector_norm(position, dim=-1)
      <= args.max_initial_palm_distance
    )
    found = sensor.data.found
    if found is None or args.min_initial_contacts <= 0:
      contacts_ok = torch.ones_like(distance_ok)
    else:
      contacts_ok = (found > 0).sum(dim=-1) >= args.min_initial_contacts
    pending &= ~(distance_ok & contacts_ok)
    if not torch.any(pending):
      break
    reset_ids = pending.nonzero(as_tuple=False).squeeze(-1)
    reset_attempts[reset_ids] += 1
    raw_env.reset(env_ids=reset_ids)
  if torch.any(pending):
    raise RuntimeError(
      f"Could not obtain valid initial grasps for {int(pending.sum())} "
      f"environments after {args.max_reset_attempts} attempts."
    )

  torch.manual_seed(args.seed + 1009)
  axes = torch.randn((args.num_envs, 3), device=args.device)
  axes = torch.nn.functional.normalize(axes, dim=-1)
  target_angles = torch.linspace(
    math.radians(args.min_target_angle_deg),
    math.radians(args.max_target_angle_deg),
    args.num_envs,
    device=args.device,
  )
  half = 0.5 * target_angles
  delta = torch.cat(
    [torch.cos(half[:, None]), axes * torch.sin(half[:, None])], dim=-1
  )
  start_quat = cube.data.root_link_quat_w.clone()
  target_quat = torch.nn.functional.normalize(
    quat_mul(delta, start_quat), dim=-1
  )
  pose_command = raw_env.command_manager.get_term("pose_goal")
  pose_command.set_goal_quaternion(target_quat)

  raw_env.observation_manager.reset(all_env_ids)
  raw_env.obs_buf = raw_env.observation_manager.compute(update_history=True)
  obs = env.get_observations()

  start_angle = quat_angular_distance(start_quat, target_quat)
  best_angle = start_angle.clone()
  latest_angle = start_angle.clone()
  active = torch.ones(args.num_envs, device=args.device, dtype=torch.bool)
  ever_success = torch.zeros_like(active)
  dwell = torch.zeros(args.num_envs, device=args.device, dtype=torch.long)
  reward_sum = torch.zeros(args.num_envs, device=args.device)
  survival_steps = torch.zeros(
    args.num_envs, device=args.device, dtype=torch.long
  )
  contact_sum = torch.zeros(4, device=args.device)
  active_sample_count = torch.zeros((), device=args.device)
  termination_counts = {
    name: torch.zeros((), device=args.device)
    for name in raw_env.termination_manager.active_terms
  }

  threshold = math.radians(args.success_angle_deg)
  with torch.inference_mode():
    for _ in range(args.num_steps):
      actions = policy(obs)
      obs, rewards, dones, _ = env.step(actions)
      dones = dones.bool()
      survival_steps += active.long()
      valid = active & ~dones
      angle = quat_angular_distance(cube.data.root_link_quat_w, target_quat)
      latest_angle = torch.where(valid, angle, latest_angle)
      best_angle = torch.where(valid, torch.minimum(best_angle, angle), best_angle)
      dwell = torch.where(
        valid & (angle < threshold), dwell + 1, torch.zeros_like(dwell)
      )
      ever_success |= dwell >= args.success_dwell_steps
      reward_sum += torch.where(active, rewards, torch.zeros_like(rewards))
      found = sensor.data.found
      if found is not None:
        contact_sum += ((found > 0) & active[:, None]).float().sum(dim=0)
        active_sample_count += active.float().sum()
      for name in termination_counts:
        termination_counts[name] += (
          raw_env.termination_manager.get_term(name) & active
        ).float().sum()
      active &= ~dones
      if not torch.any(active):
        break

  result: dict[str, object] = {
    "checkpoint": str(checkpoint),
    "num_envs": args.num_envs,
    "num_steps": args.num_steps,
    "target_angle_range_deg": [
      args.min_target_angle_deg,
      args.max_target_angle_deg,
    ],
    "success_angle_deg": args.success_angle_deg,
    "success_dwell_steps": args.success_dwell_steps,
    "success_rate": float(ever_success.float().mean()),
    "retention_rate": float(active.float().mean()),
    "mean_survival_steps": float(survival_steps.float().mean()),
    "mean_start_angle_deg": float(torch.rad2deg(start_angle).mean()),
    "mean_best_angle_deg": float(torch.rad2deg(best_angle).mean()),
    "mean_latest_angle_deg": float(torch.rad2deg(latest_angle).mean()),
    "mean_best_improvement_deg": float(
      torch.rad2deg(start_angle - best_angle).mean()
    ),
    "mean_reward_per_step": float((reward_sum / args.num_steps).mean()),
    "mean_reset_attempts": float(reset_attempts.float().mean()),
    "fingertip_contact_rate": [
      float(x)
      for x in (contact_sum / torch.clamp(active_sample_count, min=1.0)).cpu()
    ],
    "termination_counts": {
      name: int(value) for name, value in termination_counts.items()
    },
  }
  bin_edges = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0001)
  angle_deg = torch.rad2deg(start_angle)
  for low, high in zip(bin_edges[:-1], bin_edges[1:], strict=True):
    mask = (angle_deg >= low) & (angle_deg < high)
    if torch.any(mask):
      result[f"success_{int(low)}_{int(high)}_deg"] = float(
        ever_success[mask].float().mean()
      )

  print(json.dumps(result, indent=2, sort_keys=True))
  if args.json_output:
    output = Path(args.json_output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"saved: {output}")
  env.close()


if __name__ == "__main__":
  main()

