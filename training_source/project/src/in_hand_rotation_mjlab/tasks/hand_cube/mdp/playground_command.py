"""Playground's post-success moving orientation goal for cube reorientation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.utils.lab_api.math import quat_mul

from .commands import CubePoseGoalCommand, CubePoseGoalCommandCfg


class PlaygroundCubePoseGoalCommand(CubePoseGoalCommand):
  """Move the goal smoothly after success instead of jumping to a new pose.

  This follows MuJoCo Playground's task logic: a successful goal injects an
  angular goal velocity, that velocity decays each subsequent policy step, and
  the target quaternion is integrated continuously. Externally supplied play
  or hardware goals remain fixed.
  """

  cfg: "PlaygroundCubePoseGoalCommandCfg"

  def __init__(self, cfg: "PlaygroundCubePoseGoalCommandCfg", env):
    super().__init__(cfg, env)
    self._goal_velocity = torch.zeros((self.num_envs, 3), device=self.device)

  def sync_goal_marker(self) -> None:
    """Mirror the command into the optional play-only mocap target cube."""
    try:
      marker = self._env.scene["goal_cube"]
    except KeyError:
      return
    mocap_id = marker.indexing.mocap_id
    if mocap_id is None:
      raise RuntimeError("The goal_cube entity must be a mocap body.")
    self._env.sim.data.mocap_quat[:, mocap_id] = self.target_quat_w

  def set_goal_quaternion(
    self,
    goal_quat_w: torch.Tensor,
    env_ids: torch.Tensor | None = None,
  ) -> None:
    super().set_goal_quaternion(goal_quat_w, env_ids)
    self.sync_goal_marker()

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    self.sync_goal_marker()

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    assert isinstance(env_ids, torch.Tensor)
    self._goal_velocity[env_ids] = 0.0
    return super().reset(env_ids)

  def _update_metrics(self) -> None:
    cube_quat = torch.nn.functional.normalize(
      self.object.data.root_link_quat_w, dim=-1
    )
    target_quat = torch.nn.functional.normalize(self.target_quat_w, dim=-1)
    dot = torch.sum(cube_quat * target_quat, dim=-1).abs()
    angle = 2.0 * torch.acos(torch.clamp(dot, max=1.0))
    at_goal = angle < self.cfg.success_threshold
    self.success_dwell = torch.where(
      at_goal, self.success_dwell + 1, torch.zeros_like(self.success_dwell)
    )
    completed = self.success_dwell == self.cfg.success_dwell_steps
    self.completed_goals += completed.long()
    self.metrics["pose_angle"][:] = angle
    self.metrics["at_goal"][:] = at_goal.float()
    self.metrics["completed_goals"][:] = self.completed_goals.float()

    active = ~self.external_goal
    sampled_velocity = torch.empty_like(self._goal_velocity).uniform_(
      self.cfg.goal_velocity_range[0], self.cfg.goal_velocity_range[1]
    )
    inject = at_goal[:, None] & active[:, None]
    self._goal_velocity = torch.where(
      inject,
      sampled_velocity,
      self._goal_velocity * self.cfg.goal_velocity_decay,
    )
    self._goal_velocity = torch.where(
      active[:, None], self._goal_velocity, torch.zeros_like(self._goal_velocity)
    )

    rotation_vector = (
      self._goal_velocity
      * self.cfg.goal_integration_dt_scale
      * self._env.step_dt
    )
    delta_angle = torch.linalg.vector_norm(rotation_vector, dim=-1)
    safe_angle = torch.clamp(delta_angle, min=1e-8)
    axis = rotation_vector / safe_angle[:, None]
    half_angle = 0.5 * delta_angle
    delta_quat = torch.cat(
      [torch.cos(half_angle[:, None]), axis * torch.sin(half_angle[:, None])],
      dim=-1,
    )
    integrated = torch.nn.functional.normalize(
      quat_mul(self.target_quat_w, delta_quat), dim=-1
    )
    move = active & (delta_angle > 1e-8)
    self.target_quat_w[:] = torch.where(
      move[:, None], integrated, self.target_quat_w
    )
    self.sync_goal_marker()


@dataclass(kw_only=True)
class PlaygroundCubePoseGoalCommandCfg(CubePoseGoalCommandCfg):
  goal_velocity_range: tuple[float, float] = (1.0, 5.0)
  goal_velocity_decay: float = 0.8
  goal_integration_dt_scale: float = 2.0

  def build(self, env) -> PlaygroundCubePoseGoalCommand:
    if not 0.0 <= self.goal_velocity_decay < 1.0:
      raise ValueError("goal_velocity_decay must be in [0, 1)")
    if self.goal_velocity_range[0] > self.goal_velocity_range[1]:
      raise ValueError("goal_velocity_range must be ordered")
    if self.goal_integration_dt_scale <= 0.0:
      raise ValueError("goal_integration_dt_scale must be positive")
    return PlaygroundCubePoseGoalCommand(self, env)
