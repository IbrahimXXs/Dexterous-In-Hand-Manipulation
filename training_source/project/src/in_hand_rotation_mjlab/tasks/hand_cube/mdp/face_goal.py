"""Observations, rewards, and metrics for numbered-cube face reorientation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul

from .face_math import (
  cube_face_normals_w,
  rotation_6d_from_quat,
  target_face_angle,
)
from .observations import palm_center_pose_w

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_HAND_CFG = SceneEntityCfg("robot", body_names=("palm",))


def encode_camera_cube_pose(
  cube_pos_w: torch.Tensor,
  cube_quat_w: torch.Tensor,
  palm_center_pos_w: torch.Tensor,
  palm_quat_w: torch.Tensor,
) -> torch.Tensor:
  """Encode camera pose as palm-relative XYZ plus continuous rotation 6D."""
  cube_quat_w = torch.nn.functional.normalize(cube_quat_w, dim=-1)
  palm_quat_w = torch.nn.functional.normalize(palm_quat_w, dim=-1)
  world_to_palm = quat_inv(palm_quat_w)
  pos_palm = quat_apply(world_to_palm, cube_pos_w - palm_center_pos_w)
  quat_palm = quat_mul(world_to_palm, cube_quat_w)
  return torch.cat([pos_palm, rotation_6d_from_quat(quat_palm)], dim=-1)


def simulated_camera_cube_pose(
  env: ManagerBasedRlEnv,
  object_name: str,
  hand_cfg: SceneEntityCfg = _DEFAULT_HAND_CFG,
  palm_center_geom_expr: str = "palm_collision_.*",
) -> torch.Tensor:
  """Simulation stand-in for the future camera pose stream (nine values)."""
  cube: Entity = env.scene[object_name]
  palm_center_w, palm_quat_w = palm_center_pose_w(
    env=env,
    hand_cfg=hand_cfg,
    palm_center_geom_expr=palm_center_geom_expr,
  )
  return encode_camera_cube_pose(
    cube.data.root_link_pos_w,
    cube.data.root_link_quat_w,
    palm_center_w,
    palm_quat_w,
  )


def encode_target_face_goal(
  cube_quat_w: torch.Tensor,
  target_face_ids: torch.Tensor,
  palm_quat_w: torch.Tensor,
) -> torch.Tensor:
  """Encode the actionable face-up error as two palm-frame unit vectors.

  The first vector is the requested cube-face normal in its current pose. The
  second is world up. Unlike a face-number one-hot, this directly tells the
  actor which way the requested face must move, matching the orientation-error
  representation used by proven LEAP reorientation policies. It is computed
  entirely from a camera cube quaternion, the requested label, and the known
  hand/camera extrinsic, so the simulation observation has a hardware-ready
  equivalent.
  """
  cube_quat_w = torch.nn.functional.normalize(cube_quat_w, dim=-1)
  palm_quat_w = torch.nn.functional.normalize(palm_quat_w, dim=-1)
  normals_w = cube_face_normals_w(cube_quat_w)
  target_face_ids = target_face_ids.to(
    device=cube_quat_w.device, dtype=torch.long
  )
  target_normal_w = torch.gather(
    normals_w,
    dim=-2,
    index=target_face_ids[..., None, None].expand(
      target_face_ids.shape + (1, 3)
    ),
  ).squeeze(-2)
  world_up_w = torch.zeros_like(target_normal_w)
  world_up_w[..., 2] = 1.0
  world_to_palm = quat_inv(palm_quat_w)
  return torch.cat(
    [
      quat_apply(world_to_palm, target_normal_w),
      quat_apply(world_to_palm, world_up_w),
    ],
    dim=-1,
  )


def simulated_camera_face_goal(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  hand_cfg: SceneEntityCfg = _DEFAULT_HAND_CFG,
  palm_center_geom_expr: str = "palm_collision_.*",
) -> torch.Tensor:
  """Simulation stand-in for the camera-derived six-value face error."""
  cube: Entity = env.scene[object_name]
  _, palm_quat_w = palm_center_pose_w(
    env=env,
    hand_cfg=hand_cfg,
    palm_center_geom_expr=palm_center_geom_expr,
  )
  return encode_target_face_goal(
    cube.data.root_link_quat_w,
    _target_face_ids(env, command_name),
    palm_quat_w,
  )


def _target_face_ids(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return command.argmax(dim=-1)


def face_alignment_reward(
  env: ManagerBasedRlEnv, command_name: str, object_name: str
) -> torch.Tensor:
  """Dense linear angle tolerance, one at goal and zero at 180 degrees.

  Cosine shaping is flat around both zero and 180 degrees. Linear angle
  shaping keeps a useful learning gradient throughout a face transition.
  """
  cube: Entity = env.scene[object_name]
  angle = target_face_angle(
    cube.data.root_link_quat_w, _target_face_ids(env, command_name)
  )
  return torch.clamp(1.0 - angle / torch.pi, min=0.0, max=1.0)


def face_precision_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  std: float = 0.20,
) -> torch.Tensor:
  """Sharp reward near the requested face-up orientation; yaw is unconstrained."""
  if std <= 0.0:
    raise ValueError(f"std must be positive, got {std}")
  cube: Entity = env.scene[object_name]
  angle = target_face_angle(
    cube.data.root_link_quat_w, _target_face_ids(env, command_name)
  )
  return torch.exp(-0.5 * (angle / std).square())


def face_success_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  threshold: float = 0.20944,
) -> torch.Tensor:
  """Binary reward when the requested face is within ``threshold`` radians."""
  cube: Entity = env.scene[object_name]
  angle = target_face_angle(
    cube.data.root_link_quat_w, _target_face_ids(env, command_name)
  )
  return (angle < threshold).float()


class face_alignment_progress:
  """Finite-difference decrease in target-face error, robust to goal changes."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.object_name = cfg.params["object_name"]
    self.command_name = cfg.params["command_name"]
    self._cube: Entity = env.scene[self.object_name]
    self._previous_angle = torch.zeros(env.num_envs, device=env.device)
    self._previous_target = torch.full(
      (env.num_envs,), -1, device=env.device, dtype=torch.long
    )
    self._initialized = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._initialized[env_ids] = False
    self._previous_target[env_ids] = -1

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    object_name: str,
    clip: float = 2.0,
  ) -> torch.Tensor:
    del command_name, object_name
    target = _target_face_ids(env, self.command_name)
    angle = target_face_angle(self._cube.data.root_link_quat_w, target)
    valid = self._initialized & (target == self._previous_target)
    progress = (self._previous_angle - angle) / max(env.step_dt, 1e-6)
    progress = torch.where(valid, progress, torch.zeros_like(progress))
    self._previous_angle[:] = angle
    self._previous_target[:] = target
    self._initialized[:] = True
    return torch.clamp(progress, min=-clip, max=clip)


def face_goal_angle_metric(
  env: ManagerBasedRlEnv, command_name: str, object_name: str
) -> torch.Tensor:
  cube: Entity = env.scene[object_name]
  return target_face_angle(
    cube.data.root_link_quat_w, _target_face_ids(env, command_name)
  )


def face_goal_success_metric(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  threshold: float = 0.20944,
) -> torch.Tensor:
  return (face_goal_angle_metric(env, command_name, object_name) < threshold).float()

def cube_palm_distance_metric(
  env: ManagerBasedRlEnv,
  object_name: str,
  hand_cfg: SceneEntityCfg = _DEFAULT_HAND_CFG,
  palm_center_geom_expr: str = "palm_collision_.*",
) -> torch.Tensor:
  pose = simulated_camera_cube_pose(
    env=env,
    object_name=object_name,
    hand_cfg=hand_cfg,
    palm_center_geom_expr=palm_center_geom_expr,
  )
  return torch.linalg.vector_norm(pose[:, :3], dim=-1)


def cube_palm_retention_reward(
  env: ManagerBasedRlEnv,
  object_name: str,
  safe_distance: float = 0.18,
  falloff_distance: float = 0.03,
  hand_cfg: SceneEntityCfg = _DEFAULT_HAND_CFG,
  palm_center_geom_expr: str = "palm_collision_.*",
) -> torch.Tensor:
  """Reward retaining the cube near the palm without fixing its orientation.

  The canonical right-hand grasps place the cube close to the palm center, so
  distances inside ``safe_distance`` receive full credit. Credit
  then decays smoothly as the cube leaves the grasp. This supplies a dense
  learning signal before the terminal fallen-object penalty is reached.
  """
  if safe_distance <= 0.0:
    raise ValueError(f"safe_distance must be positive, got {safe_distance}")
  if falloff_distance <= 0.0:
    raise ValueError(
      f"falloff_distance must be positive, got {falloff_distance}"
    )
  distance = cube_palm_distance_metric(
    env=env,
    object_name=object_name,
    hand_cfg=hand_cfg,
    palm_center_geom_expr=palm_center_geom_expr,
  )
  excess = torch.clamp(distance - safe_distance, min=0.0)
  return torch.exp(-excess / falloff_distance)
