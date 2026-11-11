#!/usr/bin/env python3
"""Run camera-conditioned policy inference with simulated joints and no motors."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from real_world.cube_pose_io import rotation_matrix_from_quaternion_wxyz
from real_world.tracked_cube_filter import (
  FilteredJsonCubePoseSource,
  RobustCubePoseFilter,
)

import mjlab.tasks  # noqa: F401
import in_hand_rotation_mjlab.tasks  # noqa: F401


def _prepare_agent_cfg(agent_cfg) -> dict:
  cfg = asdict(agent_cfg)
  for key in ("actor", "critic"):
    if cfg[key].get("class_name", "MLPModel") != "CNNModel":
      cfg[key].pop("cnn_cfg", None)
  return cfg


def _goal_quaternion(path: Path) -> np.ndarray:
  payload = json.loads(path.read_text(encoding="utf-8"))
  value = np.asarray(payload["goal_quaternion_palm_wxyz"], dtype=np.float64)
  if value.shape != (4,) or not np.all(np.isfinite(value)):
    raise ValueError("goal_quaternion_palm_wxyz must contain four finite values")
  norm = float(np.linalg.norm(value))
  if norm < 1e-8:
    raise ValueError("goal quaternion has zero norm")
  return value / norm


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint-file", required=True)
  parser.add_argument("--pose-input-file", type=Path, required=True)
  parser.add_argument(
    "--task", default="Mjlab-Leap-Right-HandCube-Playground-Reorient"
  )
  parser.add_argument("--output-jsonl", type=Path, required=True)
  parser.add_argument("--duration-s", type=float, default=30.0)
  parser.add_argument("--inference-hz", type=float, default=20.0)
  parser.add_argument("--max-pose-age-s", type=float, default=0.25)
  parser.add_argument("--pose-hold-s", type=float, default=0.50)
  parser.add_argument("--device", default=None)
  args = parser.parse_args()

  checkpoint = Path(args.checkpoint_file).expanduser().resolve()
  pose_path = args.pose_input_file.expanduser().resolve()
  output = args.output_jsonl.expanduser().resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
  if args.duration_s <= 0.0 or args.inference_hz <= 0.0:
    raise ValueError("duration-s and inference-hz must be positive")

  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(args.task)
  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  actor_terms = tuple(raw_env.observation_manager.active_terms["actor"])
  expected_terms = (
    "joint_pos",
    "joint_pos_error",
    "camera_cube_position_error",
    "camera_cube_orientation_error",
    "last_action",
  )
  if actor_terms != expected_terms:
    raw_env.close()
    raise RuntimeError(f"unexpected actor terms: {actor_terms}")
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or OnPolicyRunner
  runner = runner_cls(env, _prepare_agent_cfg(agent_cfg), device=device)
  runner.load(str(checkpoint), map_location=device)
  policy = runner.get_inference_policy(device=device)
  pose_source = FilteredJsonCubePoseSource(
    pose_path,
    max_age_s=args.max_pose_age_s,
    hold_s=args.pose_hold_s,
    pose_filter=RobustCubePoseFilter(),
  )

  output.parent.mkdir(parents=True, exist_ok=True)
  period = 1.0 / args.inference_hz
  steps = math.ceil(args.duration_s * args.inference_hz)
  obs = env.get_observations()
  if obs.shape != (1, 57):
    env.close()
    raise RuntimeError(f"expected actor observation (1, 57), got {tuple(obs.shape)}")

  with output.open("w", encoding="utf-8") as stream, torch.inference_mode():
    for step in range(steps):
      tick = time.monotonic()
      poll = pose_source.poll()
      bounded_action = torch.zeros((1, 16), device=device)
      policy_ran = False
      if poll.pose is not None:
        goal = _goal_quaternion(pose_path)
        cube_rotation = rotation_matrix_from_quaternion_wxyz(
          poll.pose.quaternion_palm_wxyz
        )
        goal_rotation = rotation_matrix_from_quaternion_wxyz(goal)
        orientation_error = (cube_rotation @ goal_rotation.T).reshape(9)[3:]
        obs[:, 32:35] = torch.as_tensor(
          -poll.pose.position_palm_m, dtype=obs.dtype, device=device
        )
        obs[:, 35:41] = torch.as_tensor(
          orientation_error, dtype=obs.dtype, device=device
        )
        if not torch.all(torch.isfinite(obs)):
          raise RuntimeError("non-finite camera-conditioned actor observation")
        bounded_action = torch.clamp(policy(obs), -1.0, 1.0)
        policy_ran = True

      next_obs, _, _, _ = env.step(bounded_action)
      record = {
        "step": step,
        "host_timestamp_s": time.time(),
        "tracking_state": poll.state,
        "tracking_message": poll.message,
        "pose_age_s": poll.age_s,
        "policy_ran": policy_ran,
        "observation_size": int(obs.shape[1]),
        "observation_finite": bool(torch.all(torch.isfinite(obs))),
        "bounded_action": bounded_action[0].cpu().tolist(),
        "action_max_abs": float(torch.max(torch.abs(bounded_action))),
      }
      stream.write(json.dumps(record, separators=(",", ":")) + "\n")
      stream.flush()
      obs = next_obs
      remaining = period - (time.monotonic() - tick)
      if remaining > 0.0:
        time.sleep(remaining)

  env.close()
  print(f"saved: {output}")


if __name__ == "__main__":
  main()
