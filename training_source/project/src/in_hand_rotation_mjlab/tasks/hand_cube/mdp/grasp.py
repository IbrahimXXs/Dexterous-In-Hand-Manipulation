"""Table pickup rewards, metrics, resets, and geometry helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_from_euler_xyz,
  sample_uniform,
)

from .observations import palm_center_pose_w
from .pose_goal import pose_goal_angle

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_CUBE_CFG = SceneEntityCfg("cube", geom_names=("cube_geom",))


def oriented_box_min_z(
  position_w: torch.Tensor,
  quaternion_w: torch.Tensor,
  half_size: torch.Tensor,
) -> torch.Tensor:
  """Return the lowest world-Z coordinate of oriented boxes.

  Args:
    position_w: Box centers with shape ``(N, 3)``.
    quaternion_w: World orientations in ``wxyz`` order with shape ``(N, 4)``.
    half_size: Local half extents with shape ``(N, 3)`` or ``(3,)``.
  """
  rotation_wb = matrix_from_quat(quaternion_w)
  if half_size.ndim == 1:
    half_size = half_size.unsqueeze(0).expand(position_w.shape[0], -1)
  vertical_extent = torch.sum(torch.abs(rotation_wb[:, 2, :]) * half_size, dim=-1)
  return position_w[:, 2] - vertical_extent


def _cube_world_geom_id(env: ManagerBasedRlEnv, cube_cfg: SceneEntityCfg) -> int:
  cube: Entity = env.scene[cube_cfg.name]
  geom_ids = cube_cfg.geom_ids
  if isinstance(geom_ids, slice):
    local_ids = list(range(cube.num_geoms))[geom_ids]
  elif isinstance(geom_ids, torch.Tensor):
    local_ids = [int(x) for x in geom_ids.tolist()]
  else:
    local_ids = [int(x) for x in geom_ids]
  if len(local_ids) != 1:
    raise ValueError(f"Expected one cube geom, got local IDs {local_ids}.")
  return int(cube.indexing.geom_ids[local_ids[0]].item())


def _env_ids_tensor(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
) -> torch.Tensor:
  if env_ids is None:
    return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  if isinstance(env_ids, slice):
    start, stop, step = env_ids.indices(env.num_envs)
    return torch.arange(start, stop, step, device=env.device, dtype=torch.long)
  return env_ids.to(device=env.device, dtype=torch.long)


def cube_bottom_clearance(
  env: ManagerBasedRlEnv,
  table_height: float = 0.0,
  cube_cfg: SceneEntityCfg = _DEFAULT_CUBE_CFG,
) -> torch.Tensor:
  """Clearance between the lowest cube corner and the table plane, in meters."""
  cube: Entity = env.scene[cube_cfg.name]
  geom_world_id = _cube_world_geom_id(env, cube_cfg)
  half_size = env.sim.model.geom_size[:, geom_world_id, :3]
  min_z = oriented_box_min_z(
    cube.data.root_link_pos_w,
    cube.data.root_link_quat_w,
    half_size,
  )
  return min_z - table_height


def fingertip_contact_count(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Number of distinct fingertip primaries touching the cube."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return (sensor.data.found > 0).float().sum(dim=-1)


def contact_gated_lift_progress(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  target_clearance: float,
  minimum_contacts: int = 2,
  table_height: float = 0.0,
  cube_cfg: SceneEntityCfg = _DEFAULT_CUBE_CFG,
) -> torch.Tensor:
  """Normalized table-clearance reward, only while a real grasp is present."""
  if target_clearance <= 0.0:
    raise ValueError("target_clearance must be positive")
  progress = torch.clamp(
    cube_bottom_clearance(env, table_height=table_height, cube_cfg=cube_cfg)
    / target_clearance,
    min=0.0,
    max=1.0,
  )
  grasping = fingertip_contact_count(env, sensor_name) >= minimum_contacts
  return progress * grasping.float()


def sustained_grasp_lift(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  target_clearance: float,
  minimum_contacts: int = 2,
  table_height: float = 0.0,
  cube_cfg: SceneEntityCfg = _DEFAULT_CUBE_CFG,
) -> torch.Tensor:
  """Binary per-step hold reward after a genuine, contact-supported pickup."""
  clearance_ok = (
    cube_bottom_clearance(env, table_height=table_height, cube_cfg=cube_cfg)
    >= target_clearance
  )
  contact_ok = fingertip_contact_count(env, sensor_name) >= minimum_contacts
  return (clearance_ok & contact_ok).float()


def grasp_success_metric(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  target_clearance: float,
  minimum_contacts: int = 2,
  table_height: float = 0.0,
  cube_cfg: SceneEntityCfg = _DEFAULT_CUBE_CFG,
) -> torch.Tensor:
  """Instantaneous grasp success used for logs and the spawn curriculum."""
  return sustained_grasp_lift(
    env,
    sensor_name=sensor_name,
    target_clearance=target_clearance,
    minimum_contacts=minimum_contacts,
    table_height=table_height,
    cube_cfg=cube_cfg,
  )


class contact_gated_palm_lift_from_reset:
  """Reward raising a cached in-palm grasp while retaining a real pinch."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.object_name = str(cfg.params["object_name"])
    self.sensor_name = str(cfg.params["sensor_name"])
    self.hand_cfg = cfg.params.get(
      "hand_cfg", SceneEntityCfg("robot", body_names=("palm",))
    )
    self.palm_center_geom_expr = str(
      cfg.params.get("palm_center_geom_expr", "palm_collision_.*")
    )
    self._initial_height = torch.zeros(env.num_envs, device=env.device)
    self._initialized = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._initialized[slice(None) if env_ids is None else env_ids] = False

  def _lift_from_reset(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    cube: Entity = env.scene[self.object_name]
    palm_pos_w, _ = palm_center_pose_w(
      env=env,
      hand_cfg=self.hand_cfg,
      palm_center_geom_expr=self.palm_center_geom_expr,
    )
    height = cube.data.root_link_pos_w[:, 2] - palm_pos_w[:, 2]
    reset_mask = (env.episode_length_buf <= 1) | (~self._initialized)
    if torch.any(reset_mask):
      self._initial_height[reset_mask] = height[reset_mask]
      self._initialized[reset_mask] = True
    return height - self._initial_height

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    object_name: str,
    sensor_name: str,
    target_lift_m: float,
    minimum_contacts: int = 2,
    hand_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("palm",)),
    palm_center_geom_expr: str = "palm_collision_.*",
  ) -> torch.Tensor:
    del object_name, sensor_name, hand_cfg, palm_center_geom_expr
    if target_lift_m <= 0.0:
      raise ValueError("target_lift_m must be positive")
    lift = torch.clamp(self._lift_from_reset(env) / target_lift_m, min=0.0, max=1.0)
    held = fingertip_contact_count(env, self.sensor_name) >= minimum_contacts
    return lift * held.float()


class contact_gated_palm_lift_yaw_success(contact_gated_palm_lift_from_reset):
  """Binary completion: lifted, at target yaw, and still held by >=2 tips."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    object_name: str,
    sensor_name: str,
    command_name: str,
    target_lift_m: float,
    yaw_threshold: float = 0.20944,
    minimum_contacts: int = 2,
    hand_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("palm",)),
    palm_center_geom_expr: str = "palm_collision_.*",
  ) -> torch.Tensor:
    del object_name, sensor_name, hand_cfg, palm_center_geom_expr
    lifted = self._lift_from_reset(env) >= target_lift_m
    held = fingertip_contact_count(env, self.sensor_name) >= minimum_contacts
    at_yaw_goal = pose_goal_angle(env, command_name, self.object_name) < yaw_threshold
    return (lifted & held & at_yaw_goal).float()


def cube_xy_distance_from_spawn(
  env: ManagerBasedRlEnv,
  nominal_spawn_xy: tuple[float, float],
  cube_cfg: SceneEntityCfg = _DEFAULT_CUBE_CFG,
) -> torch.Tensor:
  """Horizontal cube drift from its nominal per-environment spawn point."""
  cube: Entity = env.scene[cube_cfg.name]
  nominal = torch.tensor(
    nominal_spawn_xy, device=env.device, dtype=cube.data.root_link_pos_w.dtype
  )
  relative_xy = cube.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  return torch.linalg.vector_norm(relative_xy - nominal, dim=-1)


def cube_xy_drift_penalty(
  env: ManagerBasedRlEnv,
  nominal_spawn_xy: tuple[float, float],
  deadzone: float = 0.02,
  full_penalty_distance: float = 0.10,
  cube_cfg: SceneEntityCfg = _DEFAULT_CUBE_CFG,
) -> torch.Tensor:
  """Bounded XY drift cost in ``[0, 1]`` to suppress pushing/flinging."""
  if full_penalty_distance <= deadzone:
    raise ValueError("full_penalty_distance must exceed deadzone")
  drift = cube_xy_distance_from_spawn(env, nominal_spawn_xy, cube_cfg)
  return torch.clamp(
    (drift - deadzone) / (full_penalty_distance - deadzone),
    min=0.0,
    max=1.0,
  )


def cube_xy_drift_above(
  env: ManagerBasedRlEnv,
  nominal_spawn_xy: tuple[float, float],
  maximum_distance: float,
  cube_cfg: SceneEntityCfg = _DEFAULT_CUBE_CFG,
) -> torch.Tensor:
  """Terminate cubes pushed or flung outside the grasp workspace."""
  return cube_xy_distance_from_spawn(env, nominal_spawn_xy, cube_cfg) > maximum_distance


def cube_below_table(
  env: ManagerBasedRlEnv,
  minimum_center_height: float = -0.01,
  cube_cfg: SceneEntityCfg = _DEFAULT_CUBE_CFG,
) -> torch.Tensor:
  """Terminate only after the cube has actually fallen below the support plane."""
  cube: Entity = env.scene[cube_cfg.name]
  return cube.data.root_link_pos_w[:, 2] < minimum_center_height


@requires_model_fields("geom_size")
def reset_cube_on_table(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  nominal_spawn_xy: tuple[float, float],
  xy_jitter: tuple[float, float] = (-0.01, 0.01),
  yaw_range: tuple[float, float] = (-3.14159, 3.14159),
  table_height: float = 0.0,
  spawn_clearance: float = 0.0,
  scale_list: tuple[float, ...] = (0.95, 0.9, 1.0, 1.05, 1.1),
  scale_jitter: float = 0.025,
  cube_cfg: SceneEntityCfg = _DEFAULT_CUBE_CFG,
) -> None:
  """Atomically resize and place the cube with its bottom on the table.

  The atomic write avoids using stale derived geometry during MJLab's reset
  sequence. ``spawn_clearance`` is curriculum-controlled during training and is
  exactly zero in play/deployment configuration.
  """
  resolved_env_ids = _env_ids_tensor(env, env_ids)
  if resolved_env_ids.numel() == 0:
    return
  if not scale_list:
    raise ValueError("scale_list cannot be empty")

  cube: Entity = env.scene[cube_cfg.name]
  geom_world_id = _cube_world_geom_id(env, cube_cfg)
  default_half_size = env.sim.get_default_field("geom_size")[geom_world_id, :3]

  scales = torch.tensor(scale_list, device=env.device, dtype=torch.float32)
  buckets = resolved_env_ids % len(scales)
  selected_scale = scales[buckets]
  if scale_jitter > 0.0:
    selected_scale += sample_uniform(
      -scale_jitter,
      scale_jitter,
      (resolved_env_ids.numel(),),
      device=env.device,
    )
  selected_scale = torch.clamp(selected_scale, min=0.5)
  half_size = default_half_size.unsqueeze(0) * selected_scale.unsqueeze(-1)
  env.sim.model.geom_size[resolved_env_ids, geom_world_id, :3] = half_size

  count = resolved_env_ids.numel()
  jitter_x = sample_uniform(*xy_jitter, count, device=env.device)
  jitter_y = sample_uniform(*xy_jitter, count, device=env.device)
  yaw = sample_uniform(*yaw_range, count, device=env.device)
  zero = torch.zeros_like(yaw)

  pose_w = torch.zeros((count, 7), device=env.device)
  pose_w[:, :3] = env.scene.env_origins[resolved_env_ids]
  pose_w[:, 0] += nominal_spawn_xy[0] + jitter_x
  pose_w[:, 1] += nominal_spawn_xy[1] + jitter_y
  pose_w[:, 2] += table_height + half_size[:, 2] + max(0.0, spawn_clearance)
  pose_w[:, 3:7] = quat_from_euler_xyz(zero, zero, yaw)

  cube.write_root_link_pose_to_sim(pose_w, env_ids=resolved_env_ids)
  cube.write_root_link_velocity_to_sim(
    torch.zeros((count, 6), device=env.device),
    env_ids=resolved_env_ids,
  )
