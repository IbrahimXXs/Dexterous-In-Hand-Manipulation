from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  matrix_from_quat,
  quat_mul,
  sample_uniform,
  wrap_to_pi,
)

from .face_math import target_face_angle, top_face_ids

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class InHandYawCommand(CommandTerm):
  cfg: InHandYawCommandCfg

  def __init__(self, cfg: InHandYawCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.object: Entity = env.scene[cfg.entity_name]
    self.target_yaw = torch.zeros(self.num_envs, device=self.device)

    self.metrics["yaw_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_yaw.unsqueeze(-1)

  def _cube_yaw(self) -> torch.Tensor:
    _, _, yaw = euler_xyz_from_quat(self.object.data.root_link_quat_w)
    return yaw

  def _update_metrics(self) -> None:
    yaw_error = wrap_to_pi(self.target_yaw - self._cube_yaw()).abs()
    self.metrics["yaw_error"] = yaw_error
    self.metrics["at_goal"] = (yaw_error < self.cfg.success_threshold).float()

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    current_yaw = self._cube_yaw()[env_ids]
    delta_yaw = sample_uniform(
      self.cfg.delta_yaw_range[0],
      self.cfg.delta_yaw_range[1],
      (len(env_ids),),
      device=self.device,
    )
    self.target_yaw[env_ids] = wrap_to_pi(current_yaw + delta_yaw)

  def _update_command(self) -> None:
    pass


class InHandRotationDirectionCommand(CommandTerm):
  """Command that specifies a rotation direction (+1 CCW, -1 CW) for in-hand cube rotation."""

  cfg: InHandRotationDirectionCommandCfg

  def __init__(self, cfg: InHandRotationDirectionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.object: Entity = env.scene[cfg.entity_name]

    self.direction = torch.zeros(self.num_envs, device=self.device)
    self.prev_yaw = torch.zeros(self.num_envs, device=self.device)
    self.cumulative_rotation = torch.zeros(self.num_envs, device=self.device)
    self.step_delta_yaw = torch.zeros(self.num_envs, device=self.device)

    self.metrics["total_rotation"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["directed_rotation"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.direction.unsqueeze(-1)

  def _cube_yaw(self) -> torch.Tensor:
    _, _, yaw = euler_xyz_from_quat(self.object.data.root_link_quat_w)
    return yaw

  def _update_metrics(self) -> None:
    self.metrics["total_rotation"] = self.cumulative_rotation.abs()
    self.metrics["directed_rotation"] = self.cumulative_rotation * self.direction

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.direction[env_ids] = (
      2.0 * torch.randint(0, 2, (len(env_ids),), device=self.device).float() - 1.0
    )
    self.prev_yaw[env_ids] = self._cube_yaw()[env_ids]
    self.cumulative_rotation[env_ids] = 0.0

  def _update_command(self) -> None:
    current_yaw = self._cube_yaw()
    self.step_delta_yaw = wrap_to_pi(current_yaw - self.prev_yaw)
    self.cumulative_rotation += self.step_delta_yaw
    self.prev_yaw = current_yaw.clone()


class FaceUpCommand(CommandTerm):
  """One-hot command specifying which numbered cube face should point up."""

  cfg: FaceUpCommandCfg

  def __init__(self, cfg: FaceUpCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.object: Entity = env.scene[cfg.entity_name]
    self.target_face_ids = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.long
    )
    self.success_dwell = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.long
    )
    self.completed_goals = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.long
    )
    self.metrics["face_angle"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["completed_goals"] = torch.zeros(
      self.num_envs, device=self.device
    )

    angle_min, angle_max = self.cfg.target_angle_range
    if not 0.0 <= angle_min <= angle_max <= torch.pi:
      raise ValueError(
        "target_angle_range must satisfy 0 <= min <= max <= pi, got "
        f"{self.cfg.target_angle_range}"
      )

  def _raw_cube_quat_w(self) -> torch.Tensor:
    """Read the free-joint quaternion without requiring ``sim.forward()``.

    Reset events write qpos immediately, while derived xquat remains stale
    until the environment's post-reset forward pass. Command reset happens in
    between, so using xquat here can select a goal for the previous episode.
    """
    quat_ids = self.object.indexing.free_joint_q_adr[3:7]
    return self.object.data.data.qpos[:, quat_ids]

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    assert isinstance(env_ids, torch.Tensor)
    extras = super().reset(env_ids)
    self.completed_goals[env_ids] = 0
    return extras

  @property
  def command(self) -> torch.Tensor:
    return torch.nn.functional.one_hot(
      self.target_face_ids, num_classes=6
    ).float()

  def _update_metrics(self) -> None:
    angle = target_face_angle(
      self.object.data.root_link_quat_w, self.target_face_ids
    )
    at_goal = angle < self.cfg.success_threshold
    self.success_dwell = torch.where(
      at_goal, self.success_dwell + 1, torch.zeros_like(self.success_dwell)
    )
    completed = self.success_dwell >= self.cfg.success_dwell_steps
    self.completed_goals += completed.long()
    self.metrics["face_angle"][:] = angle
    self.metrics["at_goal"][:] = at_goal.float()
    self.metrics["completed_goals"][:] = self.completed_goals.float()
    if self.cfg.resample_on_success and torch.any(completed):
      self._resample(completed.nonzero(as_tuple=False).squeeze(-1))

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    cube_quat = self._raw_cube_quat_w()[env_ids]
    face_ids = torch.arange(6, device=self.device, dtype=torch.long)
    angles = target_face_angle(cube_quat[:, None, :], face_ids[None, :])
    angle_min, angle_max = self.cfg.target_angle_range
    valid = (angles >= angle_min) & (angles <= angle_max)
    if self.cfg.exclude_current_face:
      current = top_face_ids(cube_quat)
      valid.scatter_(1, current[:, None], False)

    # A narrow curriculum interval can have no candidate for a noisy pose.
    # Fall back to any non-current face instead of producing an invalid draw.
    missing = ~valid.any(dim=-1)
    if torch.any(missing):
      valid[missing] = True
      if self.cfg.exclude_current_face:
        missing_rows = missing.nonzero(as_tuple=False).squeeze(-1)
        current = top_face_ids(cube_quat[missing_rows])
        valid[missing_rows, current] = False
    sampled = torch.multinomial(valid.float(), num_samples=1).squeeze(-1)
    self.target_face_ids[env_ids] = sampled
    self.success_dwell[env_ids] = 0

  def _update_command(self) -> None:
    pass

def _quat_angular_distance(
  quat_a: torch.Tensor, quat_b: torch.Tensor
) -> torch.Tensor:
  """Shortest SO(3) angle between wxyz quaternions."""
  quat_a = torch.nn.functional.normalize(quat_a, dim=-1)
  quat_b = torch.nn.functional.normalize(quat_b, dim=-1)
  dot = torch.sum(quat_a * quat_b, dim=-1).abs()
  return 2.0 * torch.acos(torch.clamp(dot, max=1.0))


class CubePoseGoalCommand(CommandTerm):
  """Continuous labeled-cube orientation goal in the world frame.

  Unlike FaceUpCommand, this command constrains all three orientation degrees
  of freedom. Quaternion sign is ignored, but cube rotational symmetries are
  deliberately not ignored because the physical cube is labeled and the
  requested pose is unique.

  A camera/UI bridge may call set_goal_quaternion to hold an external goal.
  Without an external goal, training samples a relative axis-angle goal from
  the configured curriculum range.
  """

  cfg: CubePoseGoalCommandCfg

  def __init__(self, cfg: CubePoseGoalCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.object: Entity = env.scene[cfg.entity_name]
    self.target_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
    self.target_quat_w[:, 0] = 1.0
    self.success_dwell = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.long
    )
    self.completed_goals = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.long
    )
    self.external_goal = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
    )
    self.metrics["pose_angle"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["completed_goals"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self._validate_angle_range()

  def _validate_angle_range(self) -> None:
    angle_min, angle_max = self.cfg.target_angle_range
    if not 0.0 <= angle_min <= angle_max <= torch.pi:
      raise ValueError(
        "target_angle_range must satisfy 0 <= min <= max <= pi, got "
        f"{self.cfg.target_angle_range}"
      )

  def _raw_cube_quat_w(self) -> torch.Tensor:
    # Reset events write qpos before derived xquat is refreshed.
    quat_ids = self.object.indexing.free_joint_q_adr[3:7]
    return self.object.data.data.qpos[:, quat_ids]

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    assert isinstance(env_ids, torch.Tensor)
    extras = super().reset(env_ids)
    self.completed_goals[env_ids] = 0
    return extras

  @property
  def command(self) -> torch.Tensor:
    return self.target_quat_w

  def set_goal_quaternion(
    self,
    goal_quat_w: torch.Tensor,
    env_ids: torch.Tensor | None = None,
  ) -> None:
    """Set and hold an externally supplied world-frame wxyz goal."""
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    else:
      env_ids = env_ids.to(device=self.device, dtype=torch.long)
    goal = torch.as_tensor(goal_quat_w, device=self.device, dtype=torch.float32)
    if goal.ndim == 1:
      goal = goal.unsqueeze(0).expand(len(env_ids), -1)
    if goal.shape != (len(env_ids), 4):
      raise ValueError(
        f"Expected goal quaternion shape {(len(env_ids), 4)}, got {tuple(goal.shape)}"
      )
    if not torch.all(torch.isfinite(goal)):
      raise ValueError("Goal quaternion contains non-finite values.")
    norms = torch.linalg.vector_norm(goal, dim=-1)
    if torch.any(norms < 1e-6):
      raise ValueError("Goal quaternion norm must be non-zero.")
    self.target_quat_w[env_ids] = torch.nn.functional.normalize(goal, dim=-1)
    self.external_goal[env_ids] = True
    self.success_dwell[env_ids] = 0
    self.time_left[env_ids] = torch.inf

  def clear_external_goal(self, env_ids: torch.Tensor | None = None) -> None:
    """Return selected environments to curriculum-driven goal sampling."""
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    else:
      env_ids = env_ids.to(device=self.device, dtype=torch.long)
    self.external_goal[env_ids] = False
    self.time_left[env_ids] = 0.0

  def _update_metrics(self) -> None:
    angle = _quat_angular_distance(
      self.object.data.root_link_quat_w, self.target_quat_w
    )
    at_goal = angle < self.cfg.success_threshold
    self.success_dwell = torch.where(
      at_goal, self.success_dwell + 1, torch.zeros_like(self.success_dwell)
    )
    completed = self.success_dwell == self.cfg.success_dwell_steps
    self.completed_goals += completed.long()
    self.metrics["pose_angle"][:] = angle
    self.metrics["at_goal"][:] = at_goal.float()
    self.metrics["completed_goals"][:] = self.completed_goals.float()
    if self.cfg.resample_on_success and torch.any(completed):
      completed_ids = completed.nonzero(as_tuple=False).squeeze(-1)
      completed_ids = completed_ids[~self.external_goal[completed_ids]]
      if len(completed_ids) > 0:
        self._resample(completed_ids)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    sampled_ids = env_ids[~self.external_goal[env_ids]]
    held_ids = env_ids[self.external_goal[env_ids]]
    if len(held_ids) > 0:
      self.time_left[held_ids] = torch.inf
    if len(sampled_ids) == 0:
      return

    self._validate_angle_range()
    axis = torch.randn((len(sampled_ids), 3), device=self.device)
    axis = torch.nn.functional.normalize(axis, dim=-1)
    angle = sample_uniform(
      self.cfg.target_angle_range[0],
      self.cfg.target_angle_range[1],
      (len(sampled_ids),),
      device=self.device,
    )
    half_angle = 0.5 * angle
    delta_quat = torch.cat(
      [torch.cos(half_angle[:, None]), axis * torch.sin(half_angle[:, None])],
      dim=-1,
    )
    current_quat = torch.nn.functional.normalize(
      self._raw_cube_quat_w()[sampled_ids], dim=-1
    )
    self.target_quat_w[sampled_ids] = torch.nn.functional.normalize(
      quat_mul(delta_quat, current_quat), dim=-1
    )
    self.success_dwell[sampled_ids] = 0

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    cube_pos_w = self.object.data.root_link_pos_w
    target_rotm_w = matrix_from_quat(self.target_quat_w)
    for env_id in env_indices:
      visualizer.add_frame(
        position=cube_pos_w[env_id],
        rotation_matrix=target_rotm_w[env_id],
        scale=self.cfg.frame_scale,
        axis_radius=self.cfg.axis_radius,
        axis_colors=self.cfg.axis_colors,
        label=f"cube_target_{env_id}",
      )



class DiscreteYawPoseGoalCommand(CubePoseGoalCommand):
  """Labeled-cube pose goal obtained only by rotating about world +Z.

  The flat right-hand task uses world +Z as the palm-normal rotation axis.
  Roll, pitch, position, and the physical grasp are unchanged by command
  sampling. The command remains a full quaternion so camera pose estimation
  has the same deployment interface as the general reorientation policy.
  """

  cfg: DiscreteYawPoseGoalCommandCfg

  def __init__(self, cfg: DiscreteYawPoseGoalCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.target_delta_yaw = torch.zeros(self.num_envs, device=self.device)
    self.metrics["target_delta_yaw"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self._validate_yaw_choices()

  def _validate_yaw_choices(self) -> None:
    if not self.cfg.target_yaw_choices:
      raise ValueError("target_yaw_choices must contain at least one angle.")
    choices = torch.as_tensor(self.cfg.target_yaw_choices)
    if not torch.all(torch.isfinite(choices)):
      raise ValueError("target_yaw_choices must be finite.")
    if torch.any(choices.abs() < 1e-6) or torch.any(choices.abs() > torch.pi):
      raise ValueError("target_yaw_choices must satisfy 0 < abs(angle) <= pi.")

  def _set_relative_yaw(
    self, env_ids: torch.Tensor, delta_yaw: torch.Tensor
  ) -> None:
    half_angle = 0.5 * delta_yaw
    zeros = torch.zeros_like(half_angle)
    delta_quat = torch.stack(
      (torch.cos(half_angle), zeros, zeros, torch.sin(half_angle)), dim=-1
    )
    current_quat = torch.nn.functional.normalize(
      self._raw_cube_quat_w()[env_ids], dim=-1
    )
    self.target_quat_w[env_ids] = torch.nn.functional.normalize(
      quat_mul(delta_quat, current_quat), dim=-1
    )
    self.target_delta_yaw[env_ids] = delta_yaw
    self.metrics["target_delta_yaw"][env_ids] = delta_yaw
    self.success_dwell[env_ids] = 0

  def set_relative_yaw_degrees(
    self, angle_degrees: float, env_ids: torch.Tensor | None = None
  ) -> None:
    """Set and hold a camera/UI-requested yaw offset from the current pose."""
    if not torch.isfinite(torch.tensor(angle_degrees)):
      raise ValueError("angle_degrees must be finite.")
    if abs(angle_degrees) < 1e-6 or abs(angle_degrees) > 180.0:
      raise ValueError("angle_degrees must satisfy 0 < abs(angle) <= 180.")
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    else:
      env_ids = env_ids.to(device=self.device, dtype=torch.long)
    delta = torch.full(
      (len(env_ids),),
      torch.deg2rad(torch.tensor(angle_degrees)).item(),
      device=self.device,
    )
    self._set_relative_yaw(env_ids, delta)
    self.external_goal[env_ids] = True
    self.time_left[env_ids] = torch.inf

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    sampled_ids = env_ids[~self.external_goal[env_ids]]
    held_ids = env_ids[self.external_goal[env_ids]]
    if len(held_ids) > 0:
      self.time_left[held_ids] = torch.inf
    if len(sampled_ids) == 0:
      return
    self._validate_yaw_choices()
    choices = torch.as_tensor(
      self.cfg.target_yaw_choices, device=self.device, dtype=torch.float32
    )
    choice_ids = torch.randint(
      len(choices), (len(sampled_ids),), device=self.device
    )
    self._set_relative_yaw(sampled_ids, choices[choice_ids])


class HandCubeFrameVizCommand(CommandTerm):
  """Debug-only command term that visualizes key task coordinate frames."""

  cfg: HandCubeFrameVizCommandCfg

  def __init__(self, cfg: HandCubeFrameVizCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.hand: Entity = env.scene[cfg.hand_name]
    self.object: Entity = env.scene[cfg.object_name]

    palm_body_ids, _ = self.hand.find_bodies(cfg.palm_body_name, preserve_order=True)
    if len(palm_body_ids) != 1:
      raise ValueError(
        f"Expected exactly one palm body for pattern '{cfg.palm_body_name}', "
        f"got {len(palm_body_ids)}."
      )
    self._palm_body_id = int(palm_body_ids[0])

    palm_geom_ids, _ = self.hand.find_geoms(
      cfg.palm_center_geom_expr, preserve_order=True
    )
    self._palm_center_geom_ids = torch.tensor(
      palm_geom_ids, dtype=torch.long, device=self.device
    )

    self._command = torch.zeros((self.num_envs, 1), device=self.device)
    self._identity_rotm = torch.eye(3, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _palm_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
    palm_pos_w = self.hand.data.body_link_pos_w[:, self._palm_body_id]
    palm_quat_w = self.hand.data.body_link_quat_w[:, self._palm_body_id]
    return palm_pos_w, palm_quat_w

  def _palm_center_pos_w(self, palm_pos_w: torch.Tensor) -> torch.Tensor:
    if self._palm_center_geom_ids.numel() == 0:
      return palm_pos_w
    return self.hand.data.geom_pos_w[:, self._palm_center_geom_ids].mean(dim=1)

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    del env_ids

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    palm_pos_w, palm_quat_w = self._palm_pose_w()
    palm_center_pos_w = self._palm_center_pos_w(palm_pos_w)
    cube_pos_w = self.object.data.root_link_pos_w
    cube_quat_w = self.object.data.root_link_quat_w

    palm_rotm_w = matrix_from_quat(palm_quat_w)
    cube_rotm_w = matrix_from_quat(cube_quat_w)

    viz = self.cfg.viz
    for env_id in env_indices:
      if viz.show_world_frame:
        visualizer.add_frame(
          position=self._env.scene.env_origins[env_id],
          rotation_matrix=self._identity_rotm,
          scale=viz.world_frame_scale,
          axis_radius=viz.axis_radius,
          axis_colors=viz.world_axis_colors,
          label=f"world_{env_id}",
        )

      if viz.show_palm_body_frame:
        visualizer.add_frame(
          position=palm_pos_w[env_id],
          rotation_matrix=palm_rotm_w[env_id],
          scale=viz.palm_body_frame_scale,
          axis_radius=viz.axis_radius,
          axis_colors=viz.palm_body_axis_colors,
          label=f"palm_body_{env_id}",
        )

      if viz.show_palm_center_frame:
        visualizer.add_frame(
          position=palm_center_pos_w[env_id],
          rotation_matrix=palm_rotm_w[env_id],
          scale=viz.palm_center_frame_scale,
          axis_radius=viz.axis_radius,
          axis_colors=viz.palm_center_axis_colors,
          label=f"palm_center_{env_id}",
        )

      if viz.show_cube_frame:
        visualizer.add_frame(
          position=cube_pos_w[env_id],
          rotation_matrix=cube_rotm_w[env_id],
          scale=viz.cube_frame_scale,
          axis_radius=viz.axis_radius,
          axis_colors=viz.cube_axis_colors,
          label=f"cube_{env_id}",
        )

      if viz.show_palm_to_cube_arrow:
        visualizer.add_arrow(
          start=palm_center_pos_w[env_id],
          end=cube_pos_w[env_id],
          color=viz.palm_to_cube_arrow_color,
          width=viz.axis_radius,
          label=f"palm_to_cube_{env_id}",
        )

      if viz.show_frame_origins:
        visualizer.add_sphere(
          center=palm_center_pos_w[env_id],
          radius=viz.origin_sphere_radius,
          color=viz.palm_center_origin_color,
          label=f"palm_center_origin_{env_id}",
        )
        visualizer.add_sphere(
          center=cube_pos_w[env_id],
          radius=viz.origin_sphere_radius,
          color=viz.cube_origin_color,
          label=f"cube_origin_{env_id}",
        )


@dataclass(kw_only=True)
class InHandYawCommandCfg(CommandTermCfg):
  entity_name: str
  delta_yaw_range: tuple[float, float] = (-1.57, 1.57)
  success_threshold: float = 0.15

  def build(self, env: ManagerBasedRlEnv) -> InHandYawCommand:
    return InHandYawCommand(self, env)


@dataclass(kw_only=True)
class InHandRotationDirectionCommandCfg(CommandTermCfg):
  entity_name: str

  def build(self, env: ManagerBasedRlEnv) -> InHandRotationDirectionCommand:
    return InHandRotationDirectionCommand(self, env)


@dataclass(kw_only=True)
class FaceUpCommandCfg(CommandTermCfg):
  entity_name: str = "cube"
  success_threshold: float = 0.20944
  success_dwell_steps: int = 5
  exclude_current_face: bool = True
  resample_on_success: bool = True
  target_angle_range: tuple[float, float] = (0.0, 3.141592653589793)

  def build(self, env: ManagerBasedRlEnv) -> FaceUpCommand:
    return FaceUpCommand(self, env)

@dataclass(kw_only=True)
class CubePoseGoalCommandCfg(CommandTermCfg):
  entity_name: str = "cube"
  success_threshold: float = 0.17453292519943295
  success_dwell_steps: int = 5
  resample_on_success: bool = True
  target_angle_range: tuple[float, float] = (
    0.17453292519943295,
    0.5235987755982988,
  )
  frame_scale: float = 0.075
  axis_radius: float = 0.006
  axis_colors: tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
  ] = ((1.0, 0.45, 0.1), (0.2, 1.0, 0.7), (0.8, 0.25, 1.0))

  def build(self, env: ManagerBasedRlEnv) -> CubePoseGoalCommand:
    return CubePoseGoalCommand(self, env)



@dataclass(kw_only=True)
class DiscreteYawPoseGoalCommandCfg(CubePoseGoalCommandCfg):
  target_yaw_choices: tuple[float, ...] = (
    -1.5707963267948966,
    1.5707963267948966,
    3.141592653589793,
  )

  def build(self, env: ManagerBasedRlEnv) -> DiscreteYawPoseGoalCommand:
    return DiscreteYawPoseGoalCommand(self, env)


@dataclass(kw_only=True)
class HandCubeFrameVizCommandCfg(CommandTermCfg):
  hand_name: str = "robot"
  object_name: str = "cube"
  palm_body_name: str = "palm"
  palm_center_geom_expr: str = "palm_collision_.*"

  @dataclass
  class VizCfg:
    axis_radius: float = 0.006
    world_frame_scale: float = 0.05
    palm_body_frame_scale: float = 0.05
    palm_center_frame_scale: float = 0.075
    cube_frame_scale: float = 0.06
    origin_sphere_radius: float = 0.006

    show_world_frame: bool = True
    show_palm_body_frame: bool = True
    show_palm_center_frame: bool = True
    show_cube_frame: bool = True
    show_palm_to_cube_arrow: bool = True
    show_frame_origins: bool = True

    world_axis_colors: tuple[
      tuple[float, float, float],
      tuple[float, float, float],
      tuple[float, float, float],
    ] = ((0.8, 0.2, 0.2), (0.2, 0.8, 0.2), (0.2, 0.2, 0.8))
    palm_body_axis_colors: tuple[
      tuple[float, float, float],
      tuple[float, float, float],
      tuple[float, float, float],
    ] = ((1.0, 0.4, 0.4), (0.4, 1.0, 0.4), (0.4, 0.4, 1.0))
    palm_center_axis_colors: tuple[
      tuple[float, float, float],
      tuple[float, float, float],
      tuple[float, float, float],
    ] = ((1.0, 0.65, 0.2), (0.2, 1.0, 0.8), (0.8, 0.2, 1.0))
    cube_axis_colors: tuple[
      tuple[float, float, float],
      tuple[float, float, float],
      tuple[float, float, float],
    ] = ((1.0, 0.25, 0.25), (0.25, 1.0, 0.25), (0.25, 0.6, 1.0))
    palm_to_cube_arrow_color: tuple[float, float, float, float] = (
      1.0,
      0.85,
      0.1,
      0.9,
    )
    palm_center_origin_color: tuple[float, float, float, float] = (
      1.0,
      0.65,
      0.15,
      0.85,
    )
    cube_origin_color: tuple[float, float, float, float] = (0.95, 0.3, 0.2, 0.85)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> HandCubeFrameVizCommand:
    return HandCubeFrameVizCommand(self, env)
