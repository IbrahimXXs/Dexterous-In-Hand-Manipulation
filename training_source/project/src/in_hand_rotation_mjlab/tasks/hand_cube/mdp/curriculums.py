from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

from .numerics import finite_or_default, safe_mean_finite, sanitize_to_range

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class RewardWeightStage(TypedDict):
  step: int
  weight: float


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term weight using training-step stages.

  Stage ``step`` is specified in environment steps (``env.common_step_counter``).
  """
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    # Use >= so stage-0 applies immediately at training start.
    if env.common_step_counter >= stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight], device=env.device)


class reward_weight_by_metric_progress:
  """Smoothly update a reward weight from episode metric progress.

  Progress is normalized from a chosen episode-averaged metric:

    progress_raw = clamp((metric - progress_min) / (progress_max - progress_min), 0, 1)

  Then smoothed with EMA and mapped to a weight interval.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg, env
    self._progress_ema: torch.Tensor | None = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice,
    reward_name: str,
    metric_name: str,
    progress_min: float,
    progress_max: float,
    weight_min: float,
    weight_max: float,
    ema_alpha: float = 0.1,
    weight_lerp: float = 0.2,
    invert_metric: bool = False,
    min_steps_per_episode: int = 8,
  ) -> dict[str, torch.Tensor]:
    if progress_max <= progress_min:
      raise ValueError(
        f"progress_max ({progress_max}) must be > progress_min ({progress_min})."
      )

    ema_alpha = float(torch.clamp(torch.tensor(ema_alpha), min=0.0, max=1.0).item())
    weight_lerp = float(
      torch.clamp(torch.tensor(weight_lerp), min=0.0, max=1.0).item()
    )

    reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)

    if metric_name not in env.metrics_manager._episode_sums:
      current_weight = finite_or_default(float(reward_term_cfg.weight), weight_min)
      reward_term_cfg.weight = current_weight
      return {
        "weight": torch.tensor(current_weight, device=env.device),
        "progress_ema": torch.tensor(0.0, device=env.device),
      }

    episode_sums = env.metrics_manager._episode_sums[metric_name][env_ids]
    step_counts = env.metrics_manager._step_count[env_ids].float()
    valid_mask = step_counts >= float(min_steps_per_episode)
    if not torch.any(valid_mask):
      current_weight = finite_or_default(float(reward_term_cfg.weight), weight_min)
      reward_term_cfg.weight = current_weight
      return {
        "weight": torch.tensor(current_weight, device=env.device),
        "progress_ema": (
          self._progress_ema
          if self._progress_ema is not None
          else torch.tensor(0.0, device=env.device)
        ),
      }

    metric_samples = episode_sums[valid_mask] / torch.clamp(
      step_counts[valid_mask], min=1.0
    )
    metric_avg = safe_mean_finite(metric_samples)
    if metric_avg is None:
      current_weight = finite_or_default(float(reward_term_cfg.weight), weight_min)
      reward_term_cfg.weight = current_weight
      return {
        "weight": torch.tensor(current_weight, device=env.device),
        "progress_ema": (
          self._progress_ema
          if self._progress_ema is not None
          else torch.tensor(0.0, device=env.device)
        ),
      }

    progress_raw = sanitize_to_range(
      (metric_avg - progress_min) / (progress_max - progress_min),
      0.0,
      1.0,
      nan_default=0.0,
    )
    if invert_metric:
      progress_raw = 1.0 - progress_raw

    if self._progress_ema is None or not torch.isfinite(self._progress_ema).item():
      self._progress_ema = progress_raw.detach()
    else:
      self._progress_ema = (
        (1.0 - ema_alpha) * self._progress_ema + ema_alpha * progress_raw.detach()
      )
    self._progress_ema = sanitize_to_range(
      self._progress_ema,
      0.0,
      1.0,
      nan_default=0.0,
    )

    target_weight = sanitize_to_range(
      weight_min + self._progress_ema * (weight_max - weight_min),
      weight_min,
      weight_max,
      nan_default=weight_min,
    )
    current_weight = torch.tensor(
      finite_or_default(float(reward_term_cfg.weight), float(target_weight.item())),
      device=env.device,
    )
    new_weight = sanitize_to_range(
      (1.0 - weight_lerp) * current_weight + weight_lerp * target_weight,
      weight_min,
      weight_max,
      nan_default=float(target_weight.item()),
    )
    reward_term_cfg.weight = float(new_weight.item())

    return {
      "metric": metric_avg.detach(),
      "progress_raw": progress_raw.detach(),
      "progress_ema": self._progress_ema.detach(),
      "target_weight": target_weight.detach(),
      "weight": new_weight.detach(),
    }


class event_param_by_metric_progress:
  """Smoothly update a scalar event parameter from episode metric progress."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg, env
    self._progress_ema: torch.Tensor | None = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice,
    event_name: str,
    param_name: str,
    metric_name: str,
    progress_min: float,
    progress_max: float,
    value_min: float,
    value_max: float,
    ema_alpha: float = 0.1,
    value_lerp: float = 0.2,
    invert_metric: bool = False,
    min_steps_per_episode: int = 8,
  ) -> dict[str, torch.Tensor]:
    if progress_max <= progress_min:
      raise ValueError(
        f"progress_max ({progress_max}) must be > progress_min ({progress_min})."
      )

    ema_alpha = float(torch.clamp(torch.tensor(ema_alpha), min=0.0, max=1.0).item())
    value_lerp = float(
      torch.clamp(torch.tensor(value_lerp), min=0.0, max=1.0).item()
    )

    term_cfg = env.event_manager.get_term_cfg(event_name)
    current_value = finite_or_default(
      float(term_cfg.params.get(param_name, value_min)), value_min
    )
    term_cfg.params[param_name] = current_value

    if metric_name not in env.metrics_manager._episode_sums:
      return {
        "value": torch.tensor(current_value, device=env.device),
        "progress_ema": torch.tensor(0.0, device=env.device),
      }

    episode_sums = env.metrics_manager._episode_sums[metric_name][env_ids]
    step_counts = env.metrics_manager._step_count[env_ids].float()
    valid_mask = step_counts >= float(min_steps_per_episode)
    if not torch.any(valid_mask):
      return {
        "value": torch.tensor(current_value, device=env.device),
        "progress_ema": (
          self._progress_ema
          if self._progress_ema is not None
          else torch.tensor(0.0, device=env.device)
        ),
      }

    metric_samples = episode_sums[valid_mask] / torch.clamp(
      step_counts[valid_mask], min=1.0
    )
    metric_avg = safe_mean_finite(metric_samples)
    if metric_avg is None:
      return {
        "value": torch.tensor(current_value, device=env.device),
        "progress_ema": (
          self._progress_ema
          if self._progress_ema is not None
          else torch.tensor(0.0, device=env.device)
        ),
      }

    progress_raw = sanitize_to_range(
      (metric_avg - progress_min) / (progress_max - progress_min),
      0.0,
      1.0,
      nan_default=0.0,
    )
    if invert_metric:
      progress_raw = 1.0 - progress_raw

    if self._progress_ema is None or not torch.isfinite(self._progress_ema).item():
      self._progress_ema = progress_raw.detach()
    else:
      self._progress_ema = (
        (1.0 - ema_alpha) * self._progress_ema + ema_alpha * progress_raw.detach()
      )
    self._progress_ema = sanitize_to_range(
      self._progress_ema,
      0.0,
      1.0,
      nan_default=0.0,
    )

    target_value = sanitize_to_range(
      value_min + self._progress_ema * (value_max - value_min),
      value_min,
      value_max,
      nan_default=value_min,
    )
    current_value_t = torch.tensor(current_value, device=env.device)
    new_value = sanitize_to_range(
      (1.0 - value_lerp) * current_value_t + value_lerp * target_value,
      value_min,
      value_max,
      nan_default=float(target_value.item()),
    )
    term_cfg.params[param_name] = float(new_value.item())

    return {
      "metric": metric_avg.detach(),
      "progress_raw": progress_raw.detach(),
      "progress_ema": self._progress_ema.detach(),
      "target_value": target_value.detach(),
      "value": new_value.detach(),
    }


class PoseGoalAngleStage(TypedDict):
  step: int
  angle_min: float
  angle_max: float


def pose_goal_angle_range(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  angle_stages: list[PoseGoalAngleStage],
) -> dict[str, torch.Tensor]:
  """Expand the sampled SO(3) target distance using environment-step stages."""
  del env_ids
  if not angle_stages:
    raise ValueError("angle_stages must contain at least one stage.")

  selected = angle_stages[0]
  for stage in angle_stages:
    if env.common_step_counter >= stage["step"]:
      selected = stage

  angle_min = float(selected["angle_min"])
  angle_max = float(selected["angle_max"])
  if not 0.0 <= angle_min <= angle_max <= torch.pi:
    raise ValueError(
      "Pose-goal curriculum angles must satisfy 0 <= min <= max <= pi, "
      f"got {(angle_min, angle_max)}."
    )
  command_term = env.command_manager.get_term(command_name)
  command_term.cfg.target_angle_range = (angle_min, angle_max)
  return {
    "angle_min_deg": torch.rad2deg(torch.tensor(angle_min, device=env.device)),
    "angle_max_deg": torch.rad2deg(torch.tensor(angle_max, device=env.device)),
  }


class PoseGoalSuccessStage(TypedDict):
  angle_min: float
  angle_max: float
  minimum_step: int
  unlock_success: float


class pose_goal_angle_by_success:
  """Advance SO(3) goal range only after the current range is being solved."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg, env
    self._stage_index = 0
    self._success_ema: torch.Tensor | None = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice,
    command_name: str,
    metric_name: str,
    stages: list[PoseGoalSuccessStage],
    ema_alpha: float = 0.05,
    min_steps_per_episode: int = 40,
  ) -> dict[str, torch.Tensor]:
    if not stages:
      raise ValueError("Pose-goal success curriculum needs at least one stage.")
    if not 0.0 < ema_alpha <= 1.0:
      raise ValueError("ema_alpha must lie in (0, 1].")
    for stage in stages:
      angle_min = float(stage["angle_min"])
      angle_max = float(stage["angle_max"])
      if not 0.0 <= angle_min <= angle_max <= torch.pi:
        raise ValueError(
          "Pose-goal curriculum angles must satisfy 0 <= min <= max <= pi, "
          f"got {(angle_min, angle_max)}."
        )
      if not 0.0 <= float(stage["unlock_success"]) <= 1.0:
        raise ValueError("unlock_success must lie in [0, 1].")

    metric_value: torch.Tensor | None = None
    if metric_name in env.metrics_manager._episode_sums:
      sums = env.metrics_manager._episode_sums[metric_name][env_ids]
      counts = env.metrics_manager._step_count[env_ids].float()
      valid = counts >= float(min_steps_per_episode)
      if torch.any(valid):
        metric_value = safe_mean_finite(
          sums[valid] / torch.clamp(counts[valid], min=1.0)
        )

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
      # Require evidence on the newly expanded range before another advance.
      self._success_ema = torch.zeros((), device=env.device)
      current = stages[self._stage_index]

    angle_min = float(current["angle_min"])
    angle_max = float(current["angle_max"])
    command_term = env.command_manager.get_term(command_name)
    command_term.cfg.target_angle_range = (angle_min, angle_max)
    return {
      "stage": torch.tensor(float(self._stage_index), device=env.device),
      "success_ema": self._success_ema,
      "angle_min_deg": torch.rad2deg(torch.tensor(angle_min, device=env.device)),
      "angle_max_deg": torch.rad2deg(torch.tensor(angle_max, device=env.device)),
    }
