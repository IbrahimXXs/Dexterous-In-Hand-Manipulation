"""MuJoCo Playground-compatible terms on the identified right LEAP model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_inv, quat_mul

from .observations import cube_pose_in_palm_frame, joint_pos_command_error
from .pose_goal import _target_quat_w

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_HAND_CFG = SceneEntityCfg("robot", body_names=("palm",))
_DEFAULT_JOINT_CFG = SceneEntityCfg("robot", joint_names=(".*",))


def playground_joint_position_error(
  env: ManagerBasedRlEnv,
  action_name: str = "joint_pos",
  biased: bool = True,
  asset_cfg: SceneEntityCfg = _DEFAULT_JOINT_CFG,
) -> torch.Tensor:
  """Measured minus commanded joint position, matching Playground's sign."""
  return -joint_pos_command_error(
    env=env,
    action_name=action_name,
    biased=biased,
    asset_cfg=asset_cfg,
  )


def playground_cube_position_error(
  env: ManagerBasedRlEnv,
  object_name: str,
  hand_cfg: SceneEntityCfg = _DEFAULT_HAND_CFG,
  palm_center_geom_expr: str = "palm_collision_.*",
) -> torch.Tensor:
  """Palm minus cube position expressed in the palm frame."""
  pose = cube_pose_in_palm_frame(
    env=env,
    object_name=object_name,
    hand_cfg=hand_cfg,
    palm_center_geom_expr=palm_center_geom_expr,
  )
  return -pose[:, :3]


def playground_cube_orientation_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  hand_cfg: SceneEntityCfg = _DEFAULT_HAND_CFG,
  palm_center_geom_expr: str = "palm_collision_.*",
) -> torch.Tensor:
  """Six entries from row-major ``R_cube @ R_goal.T`` in the palm frame."""
  cube_pose = cube_pose_in_palm_frame(
    env=env,
    object_name=object_name,
    hand_cfg=hand_cfg,
    palm_center_geom_expr=palm_center_geom_expr,
  )
  cube_quat_palm = cube_pose[:, 3:7]
  hand: Entity = env.scene[hand_cfg.name]
  palm_quat_w = hand.data.body_link_quat_w[:, hand_cfg.body_ids].squeeze(1)
  goal_quat_palm = quat_mul(
    quat_inv(palm_quat_w), _target_quat_w(env, command_name)
  )
  relative_quat = quat_mul(cube_quat_palm, quat_inv(goal_quat_palm))
  relative_matrix = matrix_from_quat(relative_quat)
  return relative_matrix.reshape(relative_matrix.shape[0], 9)[:, 3:]


def playground_orientation_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  bound: float = 0.2,
  margin: float = torch.pi,
) -> torch.Tensor:
  """Linear tolerance used by Playground for geodesic orientation error."""
  from .pose_goal import pose_goal_angle

  angle = pose_goal_angle(env, command_name, object_name)
  excess = torch.clamp(angle - bound, min=0.0)
  return torch.clamp(1.0 - excess / margin, min=0.0, max=1.0)


class playground_position_reward:
  """Reward retaining the cube near its valid reset grasp position."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._reference = torch.zeros((env.num_envs, 3), device=env.device)
    self._initialized = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._initialized[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    object_name: str,
    hand_cfg: SceneEntityCfg = _DEFAULT_HAND_CFG,
    palm_center_geom_expr: str = "palm_collision_.*",
    bound: float = 0.02,
    margin: float = 0.05,
  ) -> torch.Tensor:
    error = playground_cube_position_error(
      env=env,
      object_name=object_name,
      hand_cfg=hand_cfg,
      palm_center_geom_expr=palm_center_geom_expr,
    )
    new_ids = ~self._initialized
    if torch.any(new_ids):
      self._reference[new_ids] = error[new_ids]
      self._initialized[new_ids] = True
    drift = torch.linalg.vector_norm(error - self._reference, dim=-1)
    excess = torch.clamp(drift - bound, min=0.0)
    return torch.clamp(1.0 - excess / margin, min=0.0, max=1.0)


def playground_hand_pose_cost(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_JOINT_CFG,
) -> torch.Tensor:
  """Squared joint displacement from the identified default grasp."""
  hand: Entity = env.scene[asset_cfg.name]
  default = hand.data.default_joint_pos
  assert default is not None
  delta = (
    hand.data.joint_pos[:, asset_cfg.joint_ids]
    - default[:, asset_cfg.joint_ids]
  )
  return torch.sum(torch.square(delta), dim=-1)


def playground_action_rate_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
  """First- and second-order action difference cost from Playground."""
  action = env.action_manager.action
  previous = env.action_manager.prev_action
  previous_previous = env.action_manager.prev_prev_action
  first = torch.sum(torch.square(action - previous), dim=-1)
  second = torch.sum(
    torch.square(action - 2.0 * previous + previous_previous), dim=-1
  )
  return first + second


def playground_energy_cost(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_JOINT_CFG,
) -> torch.Tensor:
  """Sum of absolute joint speed times absolute actuator force."""
  hand: Entity = env.scene[asset_cfg.name]
  velocity = hand.data.joint_vel[:, asset_cfg.joint_ids]
  force = hand.data.actuator_force[:, asset_cfg.joint_ids]
  return torch.sum(torch.abs(velocity) * torch.abs(force), dim=-1)
