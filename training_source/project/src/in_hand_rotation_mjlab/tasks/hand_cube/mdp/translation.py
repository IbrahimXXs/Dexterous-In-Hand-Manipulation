"""Goal-conditioned in-hand object translation terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv, sample_uniform

from .observations import palm_center_pose_w

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_HAND_CFG = SceneEntityCfg("robot", body_names=("palm",))


class ObjectTranslationGoalCommand(CommandTerm):
  """Alternate a held object between two palm-X targets around its reset pose."""

  cfg: "ObjectTranslationGoalCommandCfg"

  def __init__(self, cfg: "ObjectTranslationGoalCommandCfg", env):
    super().__init__(cfg, env)
    self.object: Entity = env.scene[cfg.entity_name]
    self.hand: Entity = env.scene[cfg.hand_name]
    palm_ids, _ = self.hand.find_bodies(cfg.palm_body_name, preserve_order=True)
    if len(palm_ids) != 1:
      raise ValueError(
        f"Expected one palm body matching {cfg.palm_body_name!r}, got {palm_ids}."
      )
    self._palm_body_id = int(palm_ids[0])
    self.target_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
    self.anchor_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
    self.direction = torch.zeros(self.num_envs, device=self.device)
    self.anchor_valid = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.success_dwell = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.completed_goals = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.enabled = torch.ones(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.metrics["position_error"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["completed_goals"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self._validate_ranges()

  def _validate_ranges(self) -> None:
    for name, value in (
      ("positive_distance_range", self.cfg.positive_distance_range),
      ("negative_distance_range", self.cfg.negative_distance_range),
    ):
      if not 0.0 <= value[0] <= value[1]:
        raise ValueError(f"{name} must be nonnegative and ordered, got {value}.")
    if self.cfg.success_threshold <= 0.0:
      raise ValueError("success_threshold must be positive.")
    if self.cfg.success_dwell_steps <= 0:
      raise ValueError("success_dwell_steps must be positive.")

  def _raw_object_pos_w(self) -> torch.Tensor:
    pos_ids = self.object.indexing.free_joint_q_adr[:3]
    return self.object.data.data.qpos[:, pos_ids]

  def _palm_quat_w(self) -> torch.Tensor:
    return self.hand.data.body_link_quat_w[:, self._palm_body_id]

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos_w

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    assert isinstance(env_ids, torch.Tensor)
    self.anchor_valid[env_ids] = False
    self.direction[env_ids] = 0.0
    self.success_dwell[env_ids] = 0
    self.completed_goals[env_ids] = 0
    self.enabled[env_ids] = True
    return super().reset(env_ids)

  def _update_metrics(self) -> None:
    error = torch.linalg.vector_norm(
      self.target_pos_w - self.object.data.root_link_pos_w, dim=-1
    )
    at_goal = (error < self.cfg.success_threshold) & self.enabled
    self.success_dwell = torch.where(
      at_goal, self.success_dwell + 1, torch.zeros_like(self.success_dwell)
    )
    completed = self.success_dwell == self.cfg.success_dwell_steps
    self.completed_goals += completed.long()
    self.metrics["position_error"][:] = error
    self.metrics["at_goal"][:] = at_goal.float()
    self.metrics["completed_goals"][:] = self.completed_goals.float()
    if self.cfg.resample_on_success and torch.any(completed):
      self._resample(completed.nonzero(as_tuple=False).squeeze(-1))

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._validate_ranges()
    current_pos = self._raw_object_pos_w()[env_ids]
    new_anchor = ~self.anchor_valid[env_ids]
    if torch.any(new_anchor):
      new_ids = env_ids[new_anchor]
      self.anchor_pos_w[new_ids] = current_pos[new_anchor]
      self.anchor_valid[new_ids] = True

    previous_direction = self.direction[env_ids]
    random_direction = (
      2.0
      * torch.randint(0, 2, (len(env_ids),), device=self.device).float()
      - 1.0
    )
    direction = torch.where(
      previous_direction == 0.0, random_direction, -previous_direction
    )
    self.direction[env_ids] = direction

    positive_distance = sample_uniform(
      self.cfg.positive_distance_range[0],
      self.cfg.positive_distance_range[1],
      (len(env_ids),),
      device=self.device,
    )
    negative_distance = sample_uniform(
      self.cfg.negative_distance_range[0],
      self.cfg.negative_distance_range[1],
      (len(env_ids),),
      device=self.device,
    )
    distance = torch.where(direction > 0.0, positive_distance, negative_distance)
    offset_palm = torch.zeros((len(env_ids), 3), device=self.device)
    offset_palm[:, 0] = direction * distance
    offset_world = quat_apply(self._palm_quat_w()[env_ids], offset_palm)
    self.target_pos_w[env_ids] = self.anchor_pos_w[env_ids] + offset_world
    self.success_dwell[env_ids] = 0

  def hold_at_current(self, env_ids: torch.Tensor) -> None:
    """Freeze the goal at the current object pose during fixture ownership."""
    if len(env_ids) == 0:
      return
    current_pos = self._raw_object_pos_w()[env_ids]
    self.anchor_pos_w[env_ids] = current_pos
    self.target_pos_w[env_ids] = current_pos
    self.anchor_valid[env_ids] = True
    self.direction[env_ids] = 0.0
    self.success_dwell[env_ids] = 0
    self.enabled[env_ids] = False

  def restart_from_current(self, env_ids: torch.Tensor) -> None:
    """Start a fresh alternating translation sequence after reacquisition."""
    if len(env_ids) == 0:
      return
    self.anchor_valid[env_ids] = False
    self.direction[env_ids] = 0.0
    self.success_dwell[env_ids] = 0
    self.enabled[env_ids] = True
    self._resample(env_ids)

  def _update_command(self) -> None:
    pass


@dataclass(kw_only=True)
class ObjectTranslationGoalCommandCfg(CommandTermCfg):
  entity_name: str = "cube"
  hand_name: str = "robot"
  palm_body_name: str = "palm"
  positive_distance_range: tuple[float, float] = (0.004, 0.010)
  negative_distance_range: tuple[float, float] = (0.004, 0.010)
  success_threshold: float = 0.004
  success_dwell_steps: int = 5
  resample_on_success: bool = True

  def build(self, env) -> ObjectTranslationGoalCommand:
    return ObjectTranslationGoalCommand(self, env)


def translation_goal_error_palm(
  env: "ManagerBasedRlEnv",
  command_name: str,
  object_name: str,
  hand_cfg: SceneEntityCfg = _DEFAULT_HAND_CFG,
  palm_center_geom_expr: str = "palm_collision_.*",
) -> torch.Tensor:
  """Target-minus-object position expressed in the physical palm frame."""
  target_pos_w = env.command_manager.get_command(command_name)
  if target_pos_w is None or target_pos_w.shape[-1] != 3:
    raise ValueError(
      f"Command {command_name!r} must provide XYZ target positions."
    )
  obj: Entity = env.scene[object_name]
  _, palm_quat_w = palm_center_pose_w(
    env=env,
    hand_cfg=hand_cfg,
    palm_center_geom_expr=palm_center_geom_expr,
  )
  return quat_apply(
    quat_inv(palm_quat_w), target_pos_w - obj.data.root_link_pos_w
  )


def translation_goal_distance(
  env: "ManagerBasedRlEnv",
  command_name: str,
  object_name: str,
) -> torch.Tensor:
  target = env.command_manager.get_command(command_name)
  obj: Entity = env.scene[object_name]
  return torch.linalg.vector_norm(target - obj.data.root_link_pos_w, dim=-1)


def translation_goal_enabled(
  env: "ManagerBasedRlEnv", command_name: str
) -> torch.Tensor:
  """One where a verified pickup has enabled translation."""
  command = env.command_manager.get_term(command_name)
  enabled = getattr(command, "enabled", None)
  if enabled is None:
    return torch.ones(env.num_envs, device=env.device)
  return enabled.float()


def translation_alignment_reward(
  env: "ManagerBasedRlEnv",
  command_name: str,
  object_name: str,
  std: float = 0.015,
) -> torch.Tensor:
  if std <= 0.0:
    raise ValueError("std must be positive.")
  distance = translation_goal_distance(env, command_name, object_name)
  return (
    torch.exp(-0.5 * torch.square(distance / std))
    * translation_goal_enabled(env, command_name)
  )


def translation_goal_axis_error_fraction(
  env: "ManagerBasedRlEnv",
  command_name: str,
  object_name: str,
  axis: int = 0,
  scale: float = 0.005,
  max_fraction: float = 4.0,
  hand_cfg: SceneEntityCfg = _DEFAULT_HAND_CFG,
  palm_center_geom_expr: str = "palm_collision_.*",
) -> torch.Tensor:
  """Absolute palm-frame goal error on one axis, normalized by ``scale``."""
  if axis not in (0, 1, 2):
    raise ValueError("axis must be 0, 1, or 2")
  if scale <= 0.0 or max_fraction <= 0.0:
    raise ValueError("scale and max_fraction must be positive")
  error = translation_goal_error_palm(
    env=env,
    command_name=command_name,
    object_name=object_name,
    hand_cfg=hand_cfg,
    palm_center_geom_expr=palm_center_geom_expr,
  )
  return (
    torch.clamp(torch.abs(error[:, axis]) / scale, max=max_fraction)
    * translation_goal_enabled(env, command_name)
  )


def translation_goal_cross_error_fraction(
  env: "ManagerBasedRlEnv",
  command_name: str,
  object_name: str,
  scale: float = 0.005,
  max_fraction: float = 4.0,
  hand_cfg: SceneEntityCfg = _DEFAULT_HAND_CFG,
  palm_center_geom_expr: str = "palm_collision_.*",
) -> torch.Tensor:
  """Palm-YZ goal-error magnitude normalized by ``scale``."""
  if scale <= 0.0 or max_fraction <= 0.0:
    raise ValueError("scale and max_fraction must be positive")
  error = translation_goal_error_palm(
    env=env,
    command_name=command_name,
    object_name=object_name,
    hand_cfg=hand_cfg,
    palm_center_geom_expr=palm_center_geom_expr,
  )
  return (
    torch.clamp(
      torch.linalg.vector_norm(error[:, 1:], dim=-1) / scale,
      max=max_fraction,
    )
    * translation_goal_enabled(env, command_name)
  )


def translation_axis_alignment_reward(
  env: "ManagerBasedRlEnv",
  command_name: str,
  object_name: str,
  fraction_std: float = 0.35,
) -> torch.Tensor:
  """Reward normalized progress from the anchor toward either X target.

  Normalizing by the requested travel keeps the shaping equally meaningful
  while the curriculum expands from millimetres to the full 50/20 mm task.
  """
  if fraction_std <= 0.0:
    raise ValueError("fraction_std must be positive.")
  command = env.command_manager.get_term(command_name)
  obj: Entity = env.scene[object_name]
  desired = command.target_pos_w - command.anchor_pos_w
  actual = obj.data.root_link_pos_w - command.anchor_pos_w
  squared_distance = torch.clamp(
    torch.sum(desired * desired, dim=-1), min=1e-8
  )
  fraction = torch.sum(actual * desired, dim=-1) / squared_distance
  return (
    torch.exp(-0.5 * torch.square((fraction - 1.0) / fraction_std))
    * translation_goal_enabled(env, command_name)
  )


def translation_cross_axis_distance(
  env: "ManagerBasedRlEnv",
  command_name: str,
  object_name: str,
) -> torch.Tensor:
  """Return object drift perpendicular to the requested translation axis."""
  command = env.command_manager.get_term(command_name)
  obj: Entity = env.scene[object_name]
  desired = command.target_pos_w - command.anchor_pos_w
  direction = torch.nn.functional.normalize(desired, dim=-1)
  actual = obj.data.root_link_pos_w - command.anchor_pos_w
  projection = torch.sum(actual * direction, dim=-1, keepdim=True)
  return torch.linalg.vector_norm(actual - projection * direction, dim=-1)


def translation_cross_axis_fraction(
  env: "ManagerBasedRlEnv",
  command_name: str,
  object_name: str,
) -> torch.Tensor:
  """Cross-axis drift normalized by the current commanded travel."""
  command = env.command_manager.get_term(command_name)
  desired = command.target_pos_w - command.anchor_pos_w
  requested_distance = torch.clamp(
    torch.linalg.vector_norm(desired, dim=-1), min=1e-6
  )
  return (
    translation_cross_axis_distance(env, command_name, object_name)
    / requested_distance
    * translation_goal_enabled(env, command_name)
  )


def translation_excess_cross_axis_fraction(
  env: "ManagerBasedRlEnv",
  command_name: str,
  object_name: str,
  allowed_fraction: float = 0.5,
  max_excess: float = 2.0,
) -> torch.Tensor:
  """Penalize only cross-axis drift beyond a curriculum-scaled allowance."""
  if allowed_fraction < 0.0:
    raise ValueError("allowed_fraction must be non-negative.")
  if max_excess <= 0.0:
    raise ValueError("max_excess must be positive.")
  excess = torch.relu(
    translation_cross_axis_fraction(env, command_name, object_name)
    - allowed_fraction
  )
  return torch.clamp(excess, max=max_excess)


def translation_success_reward(
  env: "ManagerBasedRlEnv",
  command_name: str,
  object_name: str,
  threshold: float = 0.004,
) -> torch.Tensor:
  return (
    (translation_goal_distance(env, command_name, object_name) < threshold).float()
    * translation_goal_enabled(env, command_name)
  )


class translation_progress_reward:
  """Finite-difference reduction in goal distance, reset on target changes."""

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    del cfg
    self._previous_distance = torch.zeros(env.num_envs, device=env.device)
    self._previous_target = torch.zeros((env.num_envs, 3), device=env.device)
    self._initialized = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._initialized[env_ids] = False

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    object_name: str,
    clip: float = 0.25,
  ) -> torch.Tensor:
    target = env.command_manager.get_command(command_name)
    distance = translation_goal_distance(env, command_name, object_name)
    same_target = torch.linalg.vector_norm(
      target - self._previous_target, dim=-1
    ) < 1e-6
    valid = self._initialized & same_target
    progress = (self._previous_distance - distance) / max(env.step_dt, 1e-6)
    progress = torch.where(valid, progress, torch.zeros_like(progress))
    self._previous_distance[:] = distance
    self._previous_target[:] = target
    self._initialized[:] = True
    return (
      torch.clamp(progress, min=-clip, max=clip)
      * translation_goal_enabled(env, command_name)
    )


class translation_axis_progress_reward:
  """Signed finite-difference progress along the requested palm-X axis.

  Euclidean goal-distance progress can hide the correct learning signal when
  an initially imperfect cylinder grasp also settles under gravity. This term
  rewards the commanded axial component directly; cross-axis and orientation
  behavior remain independently constrained by their own terms.
  """

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    del cfg
    self._previous_fraction = torch.zeros(env.num_envs, device=env.device)
    self._previous_target = torch.zeros((env.num_envs, 3), device=env.device)
    self._initialized = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._initialized[env_ids] = False

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    object_name: str,
    max_rate: float = 5.0,
  ) -> torch.Tensor:
    if max_rate <= 0.0:
      raise ValueError("max_rate must be positive.")
    command = env.command_manager.get_term(command_name)
    obj: Entity = env.scene[object_name]
    desired = command.target_pos_w - command.anchor_pos_w
    actual = obj.data.root_link_pos_w - command.anchor_pos_w
    squared_distance = torch.clamp(
      torch.sum(desired * desired, dim=-1), min=1e-8
    )
    fraction = torch.sum(actual * desired, dim=-1) / squared_distance
    same_target = torch.linalg.vector_norm(
      command.target_pos_w - self._previous_target, dim=-1
    ) < 1e-6
    valid = self._initialized & same_target
    rate = (fraction - self._previous_fraction) / max(env.step_dt, 1e-6)
    rate = torch.where(valid, rate, torch.zeros_like(rate))
    self._previous_fraction[:] = fraction
    self._previous_target[:] = command.target_pos_w
    self._initialized[:] = True
    return (
      torch.clamp(rate, min=-max_rate, max=max_rate)
      * translation_goal_enabled(env, command_name)
    )


class translation_goal_completion_reward:
  """Emit a one-step pulse after the command's success dwell completes."""

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    del cfg
    self._previous_completed = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.long
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._previous_completed[env_ids] = 0

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
  ) -> torch.Tensor:
    completed = env.command_manager.get_term(command_name).completed_goals
    pulse = completed > self._previous_completed
    self._previous_completed[:] = completed
    return pulse.float()


def unsafe_translation_termination(
  env: "ManagerBasedRlEnv",
  exclude: tuple[str, ...] = ("time_out",),
) -> torch.Tensor:
  """Return one for any non-timeout termination already computed this step."""
  terminated = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  for name in env.termination_manager.active_terms:
    if name not in exclude:
      terminated |= env.termination_manager.get_term(name)
  return terminated.float()


def cylinder_axis_tilt(
  env: "ManagerBasedRlEnv",
  command_name: str,
  object_name: str,
) -> torch.Tensor:
  """Undirected angle between current and target cylinder longitudinal axes."""
  from .pose_goal import _target_quat_w

  obj: Entity = env.scene[object_name]
  local_axis = torch.zeros((env.num_envs, 3), device=env.device)
  local_axis[:, 2] = 1.0
  current_axis = quat_apply(obj.data.root_link_quat_w, local_axis)
  target_axis = quat_apply(_target_quat_w(env, command_name), local_axis)
  cosine = torch.abs(torch.sum(current_axis * target_axis, dim=-1))
  return torch.acos(torch.clamp(cosine, -1.0, 1.0))


def cylinder_axis_precision_reward(
  env: "ManagerBasedRlEnv",
  command_name: str,
  object_name: str,
  std: float = 0.15,
) -> torch.Tensor:
  if std <= 0.0:
    raise ValueError("std must be positive.")
  tilt = cylinder_axis_tilt(env, command_name, object_name)
  return torch.exp(-0.5 * torch.square(tilt / std))


def cylinder_axis_tilt_above(
  env: "ManagerBasedRlEnv",
  command_name: str,
  object_name: str,
  max_angle: float,
) -> torch.Tensor:
  if max_angle <= 0.0:
    raise ValueError("max_angle must be positive.")
  return cylinder_axis_tilt(env, command_name, object_name) > max_angle


def translation_orientation_error_above(
  env: "ManagerBasedRlEnv",
  command_name: str,
  object_name: str,
  max_angle: float,
) -> torch.Tensor:
  """Terminate translations that rotate the held object's marked frame."""
  if max_angle <= 0.0:
    raise ValueError("max_angle must be positive.")
  from .pose_goal import pose_goal_angle

  return pose_goal_angle(env, command_name, object_name) > max_angle


def translation_goal_success_metric(
  env: "ManagerBasedRlEnv",
  command_name: str,
  object_name: str,
  threshold: float = 0.004,
) -> torch.Tensor:
  return translation_success_reward(
    env, command_name, object_name, threshold=threshold
  )


class TranslationDistanceStage(TypedDict):
  minimum_step: int
  unlock_success: float
  positive_min: float
  positive_max: float
  negative_min: float
  negative_max: float


class translation_distance_by_success:
  """Expand palm-X travel only after the current distance is being solved."""

  def __init__(self, cfg, env: "ManagerBasedRlEnv"):
    del cfg, env
    self._stage_index = 0
    self._success_ema: torch.Tensor | None = None

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | slice,
    command_name: str,
    metric_name: str,
    stages: list[TranslationDistanceStage],
    ema_alpha: float = 0.05,
    min_steps_per_episode: int = 40,
  ) -> dict[str, torch.Tensor]:
    if not stages:
      raise ValueError("Translation curriculum needs at least one stage.")
    metric_value = None
    if metric_name in env.metrics_manager._episode_sums:
      sums = env.metrics_manager._episode_sums[metric_name][env_ids]
      counts = env.metrics_manager._step_count[env_ids].float()
      valid = counts >= float(min_steps_per_episode)
      if torch.any(valid):
        samples = sums[valid] / torch.clamp(counts[valid], min=1.0)
        finite = samples[torch.isfinite(samples)]
        if finite.numel() > 0:
          metric_value = finite.mean()

    if metric_value is not None:
      metric_value = torch.clamp(metric_value, 0.0, 1.0)
      if self._success_ema is None:
        self._success_ema = metric_value.detach()
      else:
        self._success_ema = (
          (1.0 - ema_alpha) * self._success_ema
          + ema_alpha * metric_value.detach()
        )
    if self._success_ema is None:
      self._success_ema = torch.zeros((), device=env.device)

    current = stages[self._stage_index]
    can_advance = (
      self._stage_index < len(stages) - 1
      and env.common_step_counter >= int(current["minimum_step"])
      and float(self._success_ema) >= float(current["unlock_success"])
    )
    if can_advance:
      self._stage_index += 1
      self._success_ema = torch.zeros((), device=env.device)
      current = stages[self._stage_index]

    command = env.command_manager.get_term(command_name)
    command.cfg.positive_distance_range = (
      float(current["positive_min"]),
      float(current["positive_max"]),
    )
    command.cfg.negative_distance_range = (
      float(current["negative_min"]),
      float(current["negative_max"]),
    )
    return {
      "stage": torch.tensor(float(self._stage_index), device=env.device),
      "success_ema": self._success_ema,
      "positive_max_mm": torch.tensor(
        1000.0 * float(current["positive_max"]), device=env.device
      ),
      "negative_max_mm": torch.tensor(
        1000.0 * float(current["negative_max"]), device=env.device
      ),
    }
