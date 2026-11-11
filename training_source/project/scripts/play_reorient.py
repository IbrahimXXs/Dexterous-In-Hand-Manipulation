#!/usr/bin/env python3
"""Interactive native viewer for arbitrary cube-orientation goals."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import math
from pathlib import Path
import sys
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from real_world.tracked_cube_filter import (
  FilteredJsonCubePoseSource,
  RobustCubePoseFilter,
)
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import combine_frame_transforms, quat_inv, quat_mul
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.base import ViewerAction, VerbosityLevel
from mjlab.viewer.native.keys import (
  KEY_F6,
  KEY_F7,
  KEY_F8,
  KEY_F9,
)

import mjlab.tasks  # noqa: F401
import in_hand_rotation_mjlab.tasks  # noqa: F401


def _prepare_agent_cfg(agent_cfg) -> dict[str, Any]:
  cfg = asdict(agent_cfg)
  for key in ("actor", "critic"):
    if cfg[key].get("class_name", "MLPModel") != "CNNModel":
      cfg[key].pop("cnn_cfg", None)
  return cfg


def _quaternion_from_rpy_deg(
  rpy_deg: tuple[float, float, float],
  *,
  device: torch.device | str,
) -> torch.Tensor:
  """Convert fixed palm-frame XYZ roll/pitch/yaw to a WXYZ quaternion."""
  roll, pitch, yaw = (math.radians(value) for value in rpy_deg)
  cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
  cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
  cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
  return torch.tensor(
    [
      cr * cp * cy + sr * sp * sy,
      sr * cp * cy - cr * sp * sy,
      cr * sp * cy + sr * cp * sy,
      cr * cp * sy - sr * sp * cy,
    ],
    dtype=torch.float32,
    device=device,
  )


def _set_absolute_goal_palm(
  raw_env: ManagerBasedRlEnv,
  goal_quat_p: torch.Tensor,
) -> None:
  """Set an absolute labeled-cube orientation expressed in the palm frame."""
  hand = raw_env.scene["robot"]
  palm_ids, _ = hand.find_bodies("palm", preserve_order=True)
  if len(palm_ids) != 1:
    raise RuntimeError(f"Expected one palm body, found {len(palm_ids)}.")
  palm_quat_w = hand.data.body_link_quat_w[:, palm_ids[0]]
  goal_quat_p = torch.as_tensor(
    goal_quat_p,
    dtype=torch.float32,
    device=raw_env.device,
  )
  if goal_quat_p.shape != (4,) or not torch.all(torch.isfinite(goal_quat_p)):
    raise ValueError("Palm-frame goal quaternion must contain four finite values.")
  norm = torch.linalg.vector_norm(goal_quat_p)
  if norm < 1e-6:
    raise ValueError("Palm-frame goal quaternion norm must be non-zero.")
  goal_quat_p = (goal_quat_p / norm).unsqueeze(0)
  command = raw_env.command_manager.get_term("pose_goal")
  command.set_goal_quaternion(quat_mul(palm_quat_w, goal_quat_p))


class ReorientGoalViewer(NativeMujocoViewer):
  """Apply goal changes on the viewer main thread through its action queue."""

  def _sync_goal_marker_from_command(self) -> None:
    """Mirror the held command into the visible mocap target cube."""
    try:
      marker = self.env.unwrapped.scene["goal_cube"]
    except KeyError:
      return
    mocap_id = marker.indexing.mocap_id
    if mocap_id is None:
      raise RuntimeError("The goal_cube entity must be a mocap body.")
    command = self.env.unwrapped.command_manager.get_term("pose_goal")
    target = command.target_quat_w
    self.env.unwrapped.sim.data.mocap_quat[:, mocap_id] = target
    if self.mjd is not None:
      self.mjd.mocap_quat[mocap_id] = target[self.env_idx].detach().cpu().numpy()

  def _set_goal_quaternion(self, goal_quat_w: torch.Tensor) -> None:
    """Set a held goal and immediately synchronize its visible marker."""
    command = self.env.unwrapped.command_manager.get_term("pose_goal")
    command.set_goal_quaternion(goal_quat_w)
    self._sync_goal_marker_from_command()

  def _update_tracked_cube(self) -> None:
    """Place the visual twin from the latest validated palm-frame packet."""
    pose_source = getattr(self, "pose_source", None)
    if pose_source is None:
      return

    result = pose_source.poll()
    previous_state = getattr(self, "_tracking_state", None)
    if result.state != previous_state:
      detail = f" ({result.message})" if result.message else ""
      print(f"[tracker] {result.state}{detail}")
      self._tracking_state = result.state

    marker = self.env.unwrapped.scene["tracked_cube"]
    mocap_id = marker.indexing.mocap_id
    if mocap_id is None:
      raise RuntimeError("The tracked_cube entity must be a mocap body.")

    sim_data = self.env.unwrapped.sim.data
    if result.pose is None:
      sim_data.mocap_pos[:, mocap_id] = torch.tensor(
        (0.0, 0.0, -1.0),
        dtype=torch.float32,
        device=self.env.device,
      )
      sim_data.mocap_quat[:, mocap_id] = torch.tensor(
        (1.0, 0.0, 0.0, 0.0),
        dtype=torch.float32,
        device=self.env.device,
      )
      return

    hand = self.env.unwrapped.scene["robot"]
    palm_ids, _ = hand.find_bodies("palm", preserve_order=True)
    if len(palm_ids) != 1:
      raise RuntimeError(f"Expected one palm body, found {len(palm_ids)}.")
    palm_pos_w = hand.data.body_link_pos_w[:, palm_ids[0]]
    palm_quat_w = hand.data.body_link_quat_w[:, palm_ids[0]]
    position_palm = torch.as_tensor(
      result.pose.position_palm_m,
      dtype=torch.float32,
      device=self.env.device,
    ).unsqueeze(0)
    quaternion_palm = torch.as_tensor(
      result.pose.quaternion_palm_wxyz,
      dtype=torch.float32,
      device=self.env.device,
    ).unsqueeze(0)
    position_w, quaternion_w = combine_frame_transforms(
      palm_pos_w,
      palm_quat_w,
      position_palm,
      quaternion_palm,
    )
    sim_data.mocap_pos[:, mocap_id] = position_w
    sim_data.mocap_quat[:, mocap_id] = quaternion_w

    goal_palm = result.pose.goal_quaternion_palm_wxyz
    if goal_palm is not None:
      goal_palm_tensor = torch.as_tensor(
        goal_palm,
        dtype=torch.float32,
        device=self.env.device,
      ).unsqueeze(0)
      goal_palm_tensor /= torch.linalg.vector_norm(
        goal_palm_tensor, dim=-1, keepdim=True
      ).clamp(min=1e-6)
      self._set_goal_quaternion(quat_mul(palm_quat_w, goal_palm_tensor))

  def sync_viewer_to_env(self) -> None:
    """Accept orientation edits made to the separate mocap goal cube."""
    if self.mjd is not None:
      try:
        marker = self.env.unwrapped.scene["goal_cube"]
      except KeyError:
        marker = None
      if marker is not None and marker.indexing.mocap_id is not None:
        command = self.env.unwrapped.command_manager.get_term("pose_goal")
        marker_quat = torch.as_tensor(
          self.mjd.mocap_quat[marker.indexing.mocap_id].copy(),
          dtype=torch.float32,
          device=self.env.device,
        )
        marker_quat /= torch.linalg.vector_norm(marker_quat).clamp(min=1e-6)
        target = command.target_quat_w[0]
        changed = torch.abs(torch.dot(marker_quat, target)) < 1.0 - 1e-7
        if bool(changed):
          self._set_goal_quaternion(marker_quat)
    super().sync_viewer_to_env()

  def sync_env_to_viewer(self) -> None:
    """Keep target and externally tracked cubes synchronized."""
    self._update_tracked_cube()
    self._sync_goal_marker_from_command()
    super().sync_env_to_viewer()

  def _handle_custom_action(self, action, payload) -> bool:
    if action == ViewerAction.CUSTOM and isinstance(payload, tuple):
      operation = payload[0]
      if operation == "goal_delta":
        self._rotate_goal_palm(int(payload[1]), float(payload[2]))
        return True
      if operation == "goal_current":
        cube = self.env.unwrapped.scene["cube"]
        self._set_goal_quaternion(cube.data.root_link_quat_w.clone())
        print("[viewer] target set to the current cube orientation")
        return True
      if operation == "goal_random":
        command = self.env.unwrapped.command_manager.get_term("pose_goal")
        random_quat = torch.randn_like(command.target_quat_w)
        random_quat = torch.nn.functional.normalize(random_quat, dim=-1)
        self._set_goal_quaternion(random_quat)
        print("[viewer] sampled and held a new random target")
        return True
    return super()._handle_custom_action(action, payload)

  def _rotate_goal_palm(self, axis_index: int, angle_rad: float) -> None:
    raw_env = self.env.unwrapped
    command = raw_env.command_manager.get_term("pose_goal")
    hand = raw_env.scene["robot"]
    palm_ids, _ = hand.find_bodies("palm", preserve_order=True)
    if len(palm_ids) != 1:
      raise RuntimeError(f"Expected one palm body, found {len(palm_ids)}.")
    palm_quat_w = hand.data.body_link_quat_w[:, palm_ids[0]]
    target_quat_p = quat_mul(quat_inv(palm_quat_w), command.target_quat_w)

    half = 0.5 * angle_rad
    delta = torch.zeros_like(target_quat_p)
    delta[:, 0] = math.cos(half)
    delta[:, axis_index + 1] = math.sin(half)
    new_target_p = quat_mul(delta, target_quat_p)
    new_target_w = quat_mul(palm_quat_w, new_target_p)
    self._set_goal_quaternion(new_target_w)
    axis_name = "XYZ"[axis_index]
    print(
      f"[viewer] target rotated {math.degrees(angle_rad):+.1f} deg "
      f"about palm {axis_name}"
    )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint-file", required=True)
  parser.add_argument(
    "--task", default="Mjlab-Leap-Right-HandCube-Playground-Reorient-Tilted"
  )
  parser.add_argument("--device", default=None)
  parser.add_argument("--goal-step-deg", type=float, default=15.0)
  parser.add_argument(
    "--initial-goal",
    choices=("current", "random"),
    default="current",
    help=(
      "Initial target behavior. 'current' waits at the measured cube pose; "
      "'random' uses the play task's sampled 10-180 degree target."
    ),
  )
  exact_goal = parser.add_mutually_exclusive_group()
  exact_goal.add_argument(
    "--goal-rpy-palm-deg",
    type=float,
    nargs=3,
    metavar=("ROLL", "PITCH", "YAW"),
    help=(
      "Absolute desired labeled-cube orientation in the palm frame, using "
      "fixed-frame XYZ roll/pitch/yaw in degrees."
    ),
  )
  exact_goal.add_argument(
    "--goal-quaternion-palm-wxyz",
    type=float,
    nargs=4,
    metavar=("W", "X", "Y", "Z"),
    help="Absolute desired labeled-cube orientation in the palm frame.",
  )
  parser.add_argument(
    "--pose-input-file",
    type=Path,
    help=(
      "Atomic palm-frame JSON packet. When set, an independent translucent "
      "MBZUAI cube mirrors the measured pose without affecting contacts."
    ),
  )
  parser.add_argument("--max-pose-age-s", type=float, default=0.25)
  parser.add_argument("--pose-hold-s", type=float, default=0.50)
  parser.add_argument("--position-filter-time-constant-s", type=float, default=0.035)
  parser.add_argument("--orientation-filter-time-constant-s", type=float, default=0.025)
  parser.add_argument("--max-tracked-linear-speed-m-s", type=float, default=2.5)
  parser.add_argument("--max-tracked-angular-speed-deg-s", type=float, default=1030.0)
  args = parser.parse_args()

  checkpoint = Path(args.checkpoint_file).expanduser().resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
  if not 0.0 < args.goal_step_deg <= 90.0:
    raise ValueError("--goal-step-deg must lie in (0, 90].")
  if args.max_pose_age_s <= 0.0 or args.pose_hold_s < args.max_pose_age_s:
    raise ValueError("--pose-hold-s must be at least --max-pose-age-s > 0.")
  if args.max_tracked_angular_speed_deg_s <= 0.0:
    raise ValueError("--max-tracked-angular-speed-deg-s must be positive.")

  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(args.task)
  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  if args.goal_rpy_palm_deg is not None:
    goal_quat_p = _quaternion_from_rpy_deg(
      tuple(args.goal_rpy_palm_deg),
      device=device,
    )
    _set_absolute_goal_palm(raw_env, goal_quat_p)
    print(
      "[viewer] exact palm-frame RPY target set to "
      f"{tuple(args.goal_rpy_palm_deg)} degrees"
    )
  elif args.goal_quaternion_palm_wxyz is not None:
    goal_quat_p = torch.tensor(
      args.goal_quaternion_palm_wxyz,
      dtype=torch.float32,
      device=device,
    )
    _set_absolute_goal_palm(raw_env, goal_quat_p)
    print(
      "[viewer] exact palm-frame WXYZ target set to "
      f"{tuple(args.goal_quaternion_palm_wxyz)}"
    )
  elif args.initial_goal == "current":
    command = raw_env.command_manager.get_term("pose_goal")
    cube = raw_env.scene["cube"]
    command.set_goal_quaternion(cube.data.root_link_quat_w.clone())
    print("[viewer] initial target set to the current cube orientation")
  else:
    print("[viewer] using the sampled random initial target")
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(args.task) or OnPolicyRunner
  runner = runner_cls(env, _prepare_agent_cfg(agent_cfg), device=device)
  runner.load(str(checkpoint), map_location=device)
  policy = runner.get_inference_policy(device=device)

  viewer: ReorientGoalViewer | None = None
  step = math.radians(args.goal_step_deg)
  key_actions = {
    # Function keys are deliberately used because M/N/R and digit keys are
    # claimed by MuJoCo's native viewer for rendering/debug controls.
    KEY_F6: ("goal_current",),
    KEY_F7: ("goal_delta", 2, -step),
    KEY_F8: ("goal_delta", 2, step),
    KEY_F9: ("goal_random",),
  }

  def key_callback(key: int) -> None:
    if viewer is not None and key in key_actions:
      viewer.request_action("REORIENT_GOAL", key_actions[key])

  print(
    "Visual-yaw goal keys: F7=-palm Z, F8=+palm Z, "
    "F6=current pose, F9=random target"
  )
  print(
    "Mouse goal: double-click the separate labeled target cube, then use "
    "Ctrl+right-drag to rotate it (F1 shows MuJoCo controls)."
  )
  print(
    "Viewer keys: Enter=reset, Space=pause, R=target/debug frame, "
    "-/+=speed"
  )
  viewer = ReorientGoalViewer(
    env,
    policy,
    key_callback=key_callback,
    verbosity=VerbosityLevel.INFO,
  )
  if args.pose_input_file is not None:
    pose_path = args.pose_input_file.expanduser().resolve()
    viewer.pose_source = FilteredJsonCubePoseSource(
      pose_path,
      max_age_s=args.max_pose_age_s,
      hold_s=args.pose_hold_s,
      pose_filter=RobustCubePoseFilter(
        position_time_constant_s=args.position_filter_time_constant_s,
        orientation_time_constant_s=args.orientation_filter_time_constant_s,
        max_linear_speed_m_s=args.max_tracked_linear_speed_m_s,
        max_angular_speed_rad_s=math.radians(
          args.max_tracked_angular_speed_deg_s
        ),
      ),
    )
    print(
      f"[tracker] mirroring {pose_path}; fresh <= {args.max_pose_age_s:.2f}s, "
      f"hold <= {args.pose_hold_s:.2f}s"
    )
  viewer.run()
  env.close()


if __name__ == "__main__":
  main()
