from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  quat_apply_inverse,
  wrap_to_pi,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def actuator_work_l2_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*",)),
  use_joint_acc: bool = False,
  max_abs_power: float | None = None,
  max_abs_torque: float | None = None,
  normalize: bool = False,
) -> torch.Tensor:
  """Penalize squared actuation work-like signal.

  By default this uses tau*qdot. For LEAP-style behavior, set
  ``use_joint_acc=True`` to use tau*qacc (their finite-difference proxy).
  Optional clipping prevents rare physics spikes from dominating rewards.
  """
  asset: Entity = env.scene[asset_cfg.name]
  tau = asset.data.actuator_force[:, asset_cfg.joint_ids]
  if max_abs_torque is not None:
    tau = torch.clamp(tau, min=-max_abs_torque, max=max_abs_torque)
  motion_term = (
    asset.data.joint_acc[:, asset_cfg.joint_ids]
    if use_joint_acc
    else asset.data.joint_vel[:, asset_cfg.joint_ids]
  )
  mech_power = torch.sum(tau * motion_term, dim=-1)
  if max_abs_power is not None:
    mech_power = torch.clamp(mech_power, min=-max_abs_power, max=max_abs_power)
  work = torch.square(mech_power)
  if normalize and max_abs_power is not None and max_abs_power > 0.0:
    work = work / (max_abs_power**2)
  return work


class object_yaw_finite_diff_clipped:
  """Yaw finite-difference reward term with optional temporal averaging.

  This computes signed yaw-rate from wrapped finite differences and can smooth
  it over the latest ``history_steps`` transitions:

    yaw_rate_t = ±mean_t-H+1..t(delta_yaw) / dt

  The sign is controlled by ``negate_yaw_rate``:
  - For left hand (clockwise rotation positive): negate_yaw_rate=True
  - For right hand (counter-clockwise rotation positive): negate_yaw_rate=False

  Optionally, the yaw reward is multiplied by a drift factor based on object
  translation and roll/pitch deviation from reset pose. This suppresses reward
  when the cube drifts via non-target motion.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.object_name = cfg.params["object_name"]
    self.history_steps = int(cfg.params.get("history_steps", 1))
    if self.history_steps < 1:
      raise ValueError(f"history_steps must be >= 1, got {self.history_steps}")
    self.prev_yaw = torch.zeros(env.num_envs, device=env.device)
    self._env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    self._delta_hist = torch.zeros(
      (env.num_envs, self.history_steps), device=env.device
    )
    self._hist_ptr = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self._hist_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self._init_pos_w = torch.zeros((env.num_envs, 3), device=env.device)
    self._init_roll = torch.zeros(env.num_envs, device=env.device)
    self._init_pitch = torch.zeros(env.num_envs, device=env.device)
    self._has_init = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self._env = env
    self.reset(None)

  def _current_yaw(self) -> torch.Tensor:
    cube: Entity = self._env.scene[self.object_name]
    _, _, yaw = euler_xyz_from_quat(cube.data.root_link_quat_w)
    return yaw

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    if env_ids is None:
      env_ids = slice(None)
    cube: Entity = self._env.scene[self.object_name]
    yaw = self._current_yaw()
    self.prev_yaw[env_ids] = yaw[env_ids]
    self._delta_hist[env_ids] = 0.0
    self._hist_ptr[env_ids] = 0
    self._hist_count[env_ids] = 0
    roll, pitch, _ = euler_xyz_from_quat(cube.data.root_link_quat_w)
    self._init_pos_w[env_ids] = cube.data.root_link_pos_w[env_ids]
    self._init_roll[env_ids] = roll[env_ids]
    self._init_pitch[env_ids] = pitch[env_ids]
    self._has_init[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    object_name: str,
    clip_min: float = -0.25,
    clip_max: float = 0.25,
    history_steps: int | None = None,
    negate_yaw_rate: bool = True,
    drift_position_threshold: float = 0.02,
    drift_tilt_threshold: float = 0.35,
    drift_inside_factor: float = 1.0,
    drift_outside_factor: float = 0.1,
    drift_mode: str = "step",
    drift_decay_k_pos: float = 30.0,
    drift_decay_k_tilt: float = 3.0,
    drift_min_factor: float = 0.1,
  ) -> torch.Tensor:
    del object_name  # Bound in __init__ for consistency across calls.
    if history_steps is not None and history_steps != self.history_steps:
      raise ValueError(
        f"history_steps={history_steps} does not match initialized value "
        f"{self.history_steps}. Set it via reward params."
      )

    yaw = self._current_yaw()
    # On first post-reset step, anchor the finite-diff reference.
    reset_mask = env.episode_length_buf <= 1
    if torch.any(reset_mask):
      self.reset(reset_mask.nonzero(as_tuple=False).squeeze(-1))

    delta_yaw = wrap_to_pi(yaw - self.prev_yaw)
    self.prev_yaw[:] = yaw

    # Rolling mean over latest history_steps finite differences.
    hist_ptr = self._hist_ptr
    self._delta_hist[self._env_ids, hist_ptr] = delta_yaw
    self._hist_ptr = (hist_ptr + 1) % self.history_steps
    self._hist_count = torch.clamp(self._hist_count + 1, max=self.history_steps)

    hist_len = torch.clamp(self._hist_count, min=1).to(delta_yaw.dtype)
    avg_delta_yaw = self._delta_hist.sum(dim=-1) / hist_len

    # Compute yaw rate with optional negation based on hand embodiment.
    # Left hand: negate_yaw_rate=True (clockwise rotation -> positive reward)
    # Right hand: negate_yaw_rate=False (counter-clockwise rotation -> positive reward)
    yaw_rate = avg_delta_yaw / max(env.step_dt, 1e-6)
    if negate_yaw_rate:
      yaw_rate = -yaw_rate
    yaw_reward = torch.clamp(yaw_rate, min=clip_min, max=clip_max)

    cube: Entity = self._env.scene[self.object_name]
    roll, pitch, _ = euler_xyz_from_quat(cube.data.root_link_quat_w)
    pos_error = torch.linalg.vector_norm(
      cube.data.root_link_pos_w - self._init_pos_w, dim=-1
    )
    roll_error = wrap_to_pi(roll - self._init_roll).abs()
    pitch_error = wrap_to_pi(pitch - self._init_pitch).abs()
    tilt_error = torch.linalg.vector_norm(
      torch.stack([roll_error, pitch_error], dim=-1), dim=-1
    )

    if drift_mode == "step":
      in_bounds = (pos_error <= drift_position_threshold) & (
        tilt_error <= drift_tilt_threshold
      )
      drift_factor = torch.where(
        in_bounds,
        torch.full_like(yaw_reward, drift_inside_factor),
        torch.full_like(yaw_reward, drift_outside_factor),
      )
    elif drift_mode == "exp":
      pos_excess = torch.clamp(pos_error - drift_position_threshold, min=0.0)
      tilt_excess = torch.clamp(tilt_error - drift_tilt_threshold, min=0.0)
      drift_factor = torch.exp(
        -drift_decay_k_pos * pos_excess - drift_decay_k_tilt * tilt_excess
      )
      drift_factor = torch.clamp(drift_factor, min=drift_min_factor, max=1.0)
    else:
      raise ValueError(
        f"Unknown drift_mode '{drift_mode}'. Expected one of: 'step', 'exp'."
      )

    drift_factor = torch.where(
      self._has_init, drift_factor, torch.ones_like(drift_factor)
    )
    return yaw_reward * drift_factor


def object_linvel_l1(
  env: ManagerBasedRlEnv,
  object_name: str,
  clip_max: float | None = None,
  normalize: bool = False,
) -> torch.Tensor:
  """LEAPSim linear velocity penalty term: ||v_obj||_1."""
  cube: Entity = env.scene[object_name]
  val = torch.norm(cube.data.root_link_lin_vel_w, p=1, dim=-1)
  if clip_max is not None:
    val = torch.clamp(val, max=clip_max)
  if normalize and clip_max is not None and clip_max > 0.0:
    val = val / clip_max
  return val


def fingertip_cube_contact_fraction(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  max_contacts: int = 3,
) -> torch.Tensor:
  """Contact reward in [0, 1], saturating at max_contacts fingertips.

  Full reward (1.0) is given when max_contacts (default 3) or more fingertips are
  in contact, discouraging the policy from needing all 4 fingertips simultaneously.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  in_contact = (sensor.data.found > 0).float()
  count = torch.sum(in_contact, dim=-1)
  return torch.clamp(count, max=max_contacts) / max_contacts


class fingertip_contact_participation:
  """Reward every fingertip contributing within a rolling control-step window.

  Requiring all fingertips to touch simultaneously would suppress the release
  and regrasp phases needed for continuous rotation. This term instead gives
  each fingertip credit after it touches the cube at least once during the
  recent window. The output is the fraction of fingertips that participated.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.window_steps = int(cfg.params.get("window_steps", 20))
    self.expected_fingertips = int(cfg.params.get("expected_fingertips", 4))
    self.participation_power = float(cfg.params.get("participation_power", 1.0))
    if self.window_steps < 1:
      raise ValueError(f"window_steps must be >= 1, got {self.window_steps}")
    if self.expected_fingertips < 1:
      raise ValueError(
        f"expected_fingertips must be >= 1, got {self.expected_fingertips}"
      )
    if self.participation_power < 1.0:
      raise ValueError(
        "participation_power must be >= 1.0, got "
        f"{self.participation_power}"
      )

    self._history = torch.zeros(
      (env.num_envs, self.window_steps, self.expected_fingertips),
      dtype=torch.bool,
      device=env.device,
    )
    self._history_ptr = 0

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._history[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    window_steps: int = 20,
    expected_fingertips: int = 4,
    participation_power: float = 1.0,
  ) -> torch.Tensor:
    # These parameters are bound and validated in __init__.
    del window_steps, expected_fingertips, participation_power
    sensor: ContactSensor = env.scene[sensor_name]
    found = sensor.data.found
    assert found is not None
    if found.shape[-1] != self.expected_fingertips:
      raise RuntimeError(
        f"Contact sensor '{sensor_name}' exposes {found.shape[-1]} primaries; "
        f"expected {self.expected_fingertips} fingertips."
      )

    self._history[:, self._history_ptr] = found > 0
    self._history_ptr = (self._history_ptr + 1) % self.window_steps
    participated = self._history.any(dim=1).float()
    participation_fraction = participated.mean(dim=-1)
    return participation_fraction.pow(self.participation_power)


class fingertip_cube_proximity:
  """Reward all fingertips remaining close to the oriented cube surface.

  Contact-only rewards are sparse and allow an unused finger to remain fully
  extended. This dense term measures each fingertip-geom origin's distance
  outside the cube's oriented bounding box, then combines the mean score with
  the worst fingertip score. The surface margin accounts for fingertip-geom
  thickness and permits release/regrasp without requiring constant contact.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.hand_cfg: SceneEntityCfg = cfg.params["hand_cfg"]
    self.cube_cfg: SceneEntityCfg = cfg.params["cube_cfg"]
    self.hand: Entity = env.scene[self.hand_cfg.name]
    self.cube: Entity = env.scene[self.cube_cfg.name]
    self.surface_margin = float(cfg.params.get("surface_margin", 0.018))
    self.distance_scale = float(cfg.params.get("distance_scale", 0.025))
    self.worst_tip_weight = float(cfg.params.get("worst_tip_weight", 0.6))
    if self.surface_margin < 0.0:
      raise ValueError(
        f"surface_margin must be non-negative, got {self.surface_margin}"
      )
    if self.distance_scale <= 0.0:
      raise ValueError(f"distance_scale must be positive, got {self.distance_scale}")
    if not 0.0 <= self.worst_tip_weight <= 1.0:
      raise ValueError(
        "worst_tip_weight must be in [0, 1], got "
        f"{self.worst_tip_weight}"
      )
    if isinstance(self.hand_cfg.geom_ids, slice):
      self.tip_geom_ids = list(range(self.hand.num_geoms))[self.hand_cfg.geom_ids]
    else:
      self.tip_geom_ids = self.hand_cfg.geom_ids
    if len(self.tip_geom_ids) != 4:
      raise ValueError(
        f"hand_cfg must resolve to four fingertip geoms, got {self.tip_geom_ids}."
      )
    if isinstance(self.cube_cfg.geom_ids, slice):
      self.cube_geom_ids = list(range(self.cube.num_geoms))[self.cube_cfg.geom_ids]
    else:
      self.cube_geom_ids = self.cube_cfg.geom_ids
    if len(self.cube_geom_ids) != 1:
      raise ValueError("cube_cfg must resolve to exactly one cube geom.")
    cube_local_geom_id = self.cube_geom_ids[0]
    self.cube_world_geom_id = int(
      self.cube.indexing.geom_ids[cube_local_geom_id].item()
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    hand_cfg: SceneEntityCfg,
    cube_cfg: SceneEntityCfg,
    surface_margin: float = 0.018,
    distance_scale: float = 0.025,
    worst_tip_weight: float = 0.6,
  ) -> torch.Tensor:
    del env, hand_cfg, cube_cfg, surface_margin, distance_scale, worst_tip_weight
    tip_pos_w = self.hand.data.geom_pos_w[:, self.tip_geom_ids]
    cube_pos_w = self.cube.data.geom_pos_w[:, self.cube_geom_ids].squeeze(1)
    cube_quat_w = self.cube.data.geom_quat_w[
      :, self.cube_geom_ids
    ].squeeze(1)
    num_tips = tip_pos_w.shape[1]
    cube_quat_expanded = cube_quat_w[:, None, :].expand(-1, num_tips, -1)
    tip_pos_cube = quat_apply_inverse(
      cube_quat_expanded,
      tip_pos_w - cube_pos_w[:, None, :],
    )

    cube_half_size = self.cube.data.model.geom_size[
      :, self.cube_world_geom_id, :3
    ]
    outside_delta = torch.clamp(
      tip_pos_cube.abs() - cube_half_size[:, None, :],
      min=0.0,
    )
    surface_distance = torch.linalg.vector_norm(outside_delta, dim=-1)
    distance_beyond_margin = torch.clamp(
      surface_distance - self.surface_margin,
      min=0.0,
    )
    proximity = torch.exp(-distance_beyond_margin / self.distance_scale)
    mean_score = proximity.mean(dim=-1)
    worst_score = proximity.amin(dim=-1)
    return (
      (1.0 - self.worst_tip_weight) * mean_score
      + self.worst_tip_weight * worst_score
    )


def non_thumb_finger_curl(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  open_angle: float = 0.45,
  engaged_angle: float = 0.90,
  worst_finger_weight: float = 0.5,
) -> torch.Tensor:
  """Reward the three long-finger PIP joints staying away from a flat posture."""
  if engaged_angle <= open_angle:
    raise ValueError(
      f"engaged_angle ({engaged_angle}) must exceed open_angle ({open_angle})."
    )
  if not 0.0 <= worst_finger_weight <= 1.0:
    raise ValueError(
      f"worst_finger_weight must be in [0, 1], got {worst_finger_weight}"
    )
  hand: Entity = env.scene[asset_cfg.name]
  pip_pos = hand.data.joint_pos[:, asset_cfg.joint_ids]
  curl_score = torch.clamp(
    (pip_pos - open_angle) / (engaged_angle - open_angle),
    min=0.0,
    max=1.0,
  )
  return (
    (1.0 - worst_finger_weight) * curl_score.mean(dim=-1)
    + worst_finger_weight * curl_score.amin(dim=-1)
  )


def object_fallen(
  env: ManagerBasedRlEnv,
  object_name: str,
  minimum_height: float,
) -> torch.Tensor:
  """Indicator that object center fell below minimum height."""
  cube: Entity = env.scene[object_name]
  return (cube.data.root_link_pos_w[:, 2] < minimum_height).float()


def joint_torque_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*",)),
  max_abs_torque: float | None = None,
  normalize: bool = False,
) -> torch.Tensor:
  """LEAPSim torque penalty term: sum(tau^2)."""
  asset: Entity = env.scene[asset_cfg.name]
  tau = asset.data.actuator_force[:, asset_cfg.joint_ids]
  if max_abs_torque is not None:
    tau = torch.clamp(tau, min=-max_abs_torque, max=max_abs_torque)
  val = torch.sum(torch.square(tau), dim=-1)
  if normalize:
    if max_abs_torque is not None and max_abs_torque > 0.0:
      denom = tau.shape[1] * (max_abs_torque**2)
    else:
      denom = max(tau.shape[1], 1)
    val = val / denom
  return val


class pose_diff_l2_from_reset:
  """Pose-diff penalty from reset with optional per-joint dead-zone."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self.asset: Entity = env.scene[self.asset_cfg.name]
    num_joints = self.asset.data.joint_pos[:, self.asset_cfg.joint_ids].shape[1]
    self.init_joint_pos = torch.zeros(
      (env.num_envs, num_joints),
      device=env.device,
      dtype=self.asset.data.joint_pos.dtype,
    )
    self.reset(None)

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    if env_ids is None:
      env_ids = slice(None)
    self.init_joint_pos[env_ids] = self.asset.data.joint_pos[env_ids][
      :, self.asset_cfg.joint_ids
    ]

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    average_per_joint: bool = True,
    joint_tolerance: float = 0.0,
    clip_max: float | None = None,
    normalize: bool = False,
  ) -> torch.Tensor:
    del env, asset_cfg
    q = self.asset.data.joint_pos[:, self.asset_cfg.joint_ids]
    joint_diff = torch.abs(q - self.init_joint_pos)
    if joint_tolerance > 0.0:
      # Ignore small deviations around reset; only penalize excess beyond tolerance.
      joint_diff = torch.clamp(joint_diff - joint_tolerance, min=0.0)
    val = torch.sum(torch.square(joint_diff), dim=-1)
    if average_per_joint:
      val = val / max(q.shape[1], 1)
    if clip_max is not None:
      val = torch.clamp(val, max=clip_max)
    if normalize and clip_max is not None and clip_max > 0.0:
      val = val / clip_max
    return val
