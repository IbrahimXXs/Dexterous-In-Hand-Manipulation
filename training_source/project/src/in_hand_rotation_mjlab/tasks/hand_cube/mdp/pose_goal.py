"""Camera-ready observations and rewards for arbitrary cube SO(3) goals."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_inv, quat_mul

from .face_goal import simulated_camera_cube_pose
from .face_math import rotation_6d_from_quat
from .observations import palm_center_pose_w

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_HAND_CFG = SceneEntityCfg("robot", body_names=("palm",))


def quat_angular_distance(
  quat_a: torch.Tensor, quat_b: torch.Tensor
) -> torch.Tensor:
  """Return the shortest physical rotation angle between wxyz quaternions."""
  quat_a = torch.nn.functional.normalize(quat_a, dim=-1)
  quat_b = torch.nn.functional.normalize(quat_b, dim=-1)
  dot = torch.sum(quat_a * quat_b, dim=-1).abs()
  return 2.0 * torch.acos(torch.clamp(dot, max=1.0))


def encode_target_orientation_palm(
  target_quat_w: torch.Tensor,
  palm_quat_w: torch.Tensor,
) -> torch.Tensor:
  """Encode a world-frame orientation target as palm-frame rotation 6D.

  This representation is continuous at quaternion sign flips and has the same
  six-value shape in simulation and from a calibrated real camera.
  """
  target_quat_w = torch.nn.functional.normalize(target_quat_w, dim=-1)
  palm_quat_w = torch.nn.functional.normalize(palm_quat_w, dim=-1)
  target_quat_palm = quat_mul(quat_inv(palm_quat_w), target_quat_w)
  return rotation_6d_from_quat(target_quat_palm)


def _target_quat_w(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  if command is None or command.shape[-1] != 4:
    raise ValueError(
      f"Command '{command_name}' must provide a wxyz quaternion, got "
      f"{None if command is None else tuple(command.shape)}"
    )
  return torch.nn.functional.normalize(command, dim=-1)


def simulated_camera_pose_goal(
  env: ManagerBasedRlEnv,
  command_name: str,
  hand_cfg: SceneEntityCfg = _DEFAULT_HAND_CFG,
  palm_center_geom_expr: str = "palm_collision_.*",
) -> torch.Tensor:
  """Simulation stand-in for the camera/UI-derived six-value target pose."""
  _, palm_quat_w = palm_center_pose_w(
    env=env,
    hand_cfg=hand_cfg,
    palm_center_geom_expr=palm_center_geom_expr,
  )
  return encode_target_orientation_palm(
    _target_quat_w(env, command_name),
    palm_quat_w,
  )


def camera_pose_and_goal_observation(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  hand_cfg: SceneEntityCfg = _DEFAULT_HAND_CFG,
  palm_center_geom_expr: str = "palm_collision_.*",
) -> torch.Tensor:
  """Return the exact 15-value exteroceptive deployment contract."""
  cube_pose = simulated_camera_cube_pose(
    env=env,
    object_name=object_name,
    hand_cfg=hand_cfg,
    palm_center_geom_expr=palm_center_geom_expr,
  )
  pose_goal = simulated_camera_pose_goal(
    env=env,
    command_name=command_name,
    hand_cfg=hand_cfg,
    palm_center_geom_expr=palm_center_geom_expr,
  )
  return torch.cat([cube_pose, pose_goal], dim=-1)


def pose_goal_angle(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
) -> torch.Tensor:
  cube: Entity = env.scene[object_name]
  return quat_angular_distance(
    cube.data.root_link_quat_w,
    _target_quat_w(env, command_name),
  )


def pose_alignment_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
) -> torch.Tensor:
  """Dense geodesic shaping with a nonzero slope over the whole SO(3) range."""
  angle = pose_goal_angle(env, command_name, object_name)
  return torch.clamp(1.0 - angle / torch.pi, min=0.0, max=1.0)


def pose_precision_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  std: float = 0.20,
) -> torch.Tensor:
  """Sharp Gaussian reward near the requested unique orientation."""
  if std <= 0.0:
    raise ValueError(f"std must be positive, got {std}")
  angle = pose_goal_angle(env, command_name, object_name)
  return torch.exp(-0.5 * (angle / std).square())


def pose_success_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  threshold: float = 0.17453292519943295,
) -> torch.Tensor:
  """Binary reward inside the requested SO(3) tolerance."""
  return (pose_goal_angle(env, command_name, object_name) < threshold).float()


class pose_alignment_progress:
  """Finite-difference decrease in pose error, robust to command changes."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.object_name = cfg.params["object_name"]
    self.command_name = cfg.params["command_name"]
    self._cube: Entity = env.scene[self.object_name]
    self._previous_angle = torch.zeros(env.num_envs, device=env.device)
    self._previous_target = torch.zeros((env.num_envs, 4), device=env.device)
    self._previous_target[:, 0] = 1.0
    self._initialized = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._initialized[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    object_name: str,
    clip: float = 2.0,
  ) -> torch.Tensor:
    del command_name, object_name
    target = _target_quat_w(env, self.command_name)
    angle = quat_angular_distance(self._cube.data.root_link_quat_w, target)
    same_target = (
      torch.sum(target * self._previous_target, dim=-1).abs() > 1.0 - 1e-5
    )
    valid = self._initialized & same_target
    progress = (self._previous_angle - angle) / max(env.step_dt, 1e-6)
    progress = torch.where(valid, progress, torch.zeros_like(progress))
    self._previous_angle[:] = angle
    self._previous_target[:] = target
    self._initialized[:] = True
    return torch.clamp(progress, min=-clip, max=clip)


def pose_goal_angle_metric(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
) -> torch.Tensor:
  return pose_goal_angle(env, command_name, object_name)


def pose_goal_success_metric(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  threshold: float = 0.17453292519943295,
) -> torch.Tensor:
  return (pose_goal_angle(env, command_name, object_name) < threshold).float()

