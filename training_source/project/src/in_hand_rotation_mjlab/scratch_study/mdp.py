"""Corrected MDP terms for the from-scratch state-estimation study.

Every term that compares the cube with its "episode-initial" state uses the
same anchoring rule: ``reset()`` only *marks* an environment, and the anchor is
taken on the term's next evaluation, i.e. after the first physics step of the
new episode.  MJLab resets managers before ``sim.forward()``, so anything read
inside ``reset()`` can still be the previous episode's final kinematics; this
rule makes stale anchors impossible (finding 1) and also covers the runner's
random initial episode lengths.  Tilt is the physical tilt of
``geometry.py`` instead of XYZ-Euler roll/pitch differences (finding 2), and
rotation is the singularity-free yaw increment of ``geometry.py``.

Thresholds, weights, clipping and gating values are unchanged from the
original task; only *what is measured* is corrected.
"""

from __future__ import annotations

import torch

from .geometry import physical_tilt, world_up_in_body, yaw_increment


# --------------------------------------------------------------------------
# anchoring helper
# --------------------------------------------------------------------------
class _CubeAnchor:
  """Episode-initial cube position / attitude, anchored on fresh kinematics."""

  def __init__(self, env, object_name: str = "cube"):
    self.cube = env.scene[object_name]
    n, dev = env.num_envs, env.device
    self.pos0 = torch.zeros(n, 3, device=dev)
    self.up0 = torch.zeros(n, 3, device=dev)
    self.up0[:, 2] = 1.0
    self.needs = torch.ones(n, dtype=torch.bool, device=dev)

  def mark(self, env_ids) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.needs[env_ids] = True

  def update(self) -> torch.Tensor:
    """Anchor marked envs from current kinematics; return the anchored mask."""
    anchored = self.needs.clone()
    if bool(anchored.any()):
      self.pos0[anchored] = self.cube.data.root_link_pos_w[anchored]
      self.up0[anchored] = world_up_in_body(self.cube.data.root_link_quat_w[anchored])
      self.needs[anchored] = False
    return anchored

  def drift(self) -> torch.Tensor:
    return torch.linalg.vector_norm(self.cube.data.root_link_pos_w - self.pos0, dim=-1)

  def tilt(self) -> torch.Tensor:
    return physical_tilt(self.cube.data.root_link_quat_w, up_ref=self.up0)


class _YawTracker:
  """Per-step genuine yaw increment; zero on the anchoring step."""

  def __init__(self, env, object_name: str = "cube"):
    self.cube = env.scene[object_name]
    self.prev_q = self.cube.data.root_link_quat_w.clone()
    self.needs = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

  def mark(self, env_ids) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.needs[env_ids] = True

  def step(self) -> torch.Tensor:
    q = self.cube.data.root_link_quat_w
    d = yaw_increment(q, self.prev_q)
    d = torch.where(self.needs, torch.zeros_like(d), d)
    self.prev_q = q.clone()
    self.needs[:] = False
    return d


# --------------------------------------------------------------------------
# termination
# --------------------------------------------------------------------------
class cube_pose_deviation_physical:
  """Terminate when the cube drifts > max_position_error (m) or tilts >
  max_tilt_error (rad, physical tilt) from its episode-initial pose."""

  def __init__(self, cfg, env):
    self.anchor = _CubeAnchor(env, cfg.params.get("object_name", "cube"))

  def reset(self, env_ids=None):
    self.anchor.mark(env_ids)

  def __call__(self, env, max_position_error: float, max_tilt_error: float, object_name: str = "cube"):
    anchored = self.anchor.update()
    bad = (self.anchor.drift() > max_position_error) | (self.anchor.tilt() > max_tilt_error)
    return bad & ~anchored


# --------------------------------------------------------------------------
# rewards
# --------------------------------------------------------------------------
class genuine_yaw_rate_reward:
  """Original drift-gated, clipped yaw-rate reward with corrected measurement.

  yaw_rate = mean of the last ``history_steps`` genuine yaw increments / dt,
  clipped to [clip_min, clip_max], multiplied by ``drift_outside_factor`` when
  drift > drift_position_threshold or physical tilt > drift_tilt_threshold.
  """

  def __init__(self, cfg, env):
    p = cfg.params
    self.h = int(p.get("history_steps", 4))
    self.anchor = _CubeAnchor(env, p.get("object_name", "cube"))
    self.yaw = _YawTracker(env, p.get("object_name", "cube"))
    self.hist = torch.zeros(env.num_envs, self.h, device=env.device)
    self.count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self.ptr = 0

  def reset(self, env_ids=None):
    self.anchor.mark(env_ids)
    self.yaw.mark(env_ids)
    if env_ids is None:
      env_ids = slice(None)
    self.hist[env_ids] = 0.0
    self.count[env_ids] = 0

  def __call__(
    self,
    env,
    object_name: str = "cube",
    clip_min: float = -0.25,
    clip_max: float = 0.25,
    history_steps: int = 4,
    drift_position_threshold: float = 0.02,
    drift_tilt_threshold: float = 0.35,
    drift_inside_factor: float = 1.0,
    drift_outside_factor: float = 0.1,
  ):
    self.anchor.update()
    d = self.yaw.step()
    self.hist[:, self.ptr] = d
    self.ptr = (self.ptr + 1) % self.h
    self.count = (self.count + 1).clamp(max=self.h)
    rate = self.hist.sum(-1) / self.count.clamp_min(1) / env.step_dt
    r = rate.clamp(clip_min, clip_max)
    inside = (self.anchor.drift() <= drift_position_threshold) & (self.anchor.tilt() <= drift_tilt_threshold)
    return r * torch.where(inside, drift_inside_factor, drift_outside_factor)


class pose_diff_l2_from_reset_anchored:
  """Original joint-pose penalty, anchored on the first fresh post-reset step."""

  def __init__(self, cfg, env):
    self.asset_cfg = cfg.params["asset_cfg"]
    self.asset = env.scene[self.asset_cfg.name]
    q = self.asset.data.joint_pos[:, self.asset_cfg.joint_ids]
    self.q0 = q.clone()
    self.needs = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids=None):
    if env_ids is None:
      env_ids = slice(None)
    self.needs[env_ids] = True

  def __call__(self, env, asset_cfg, average_per_joint: bool = False, joint_tolerance: float = 0.4):
    q = self.asset.data.joint_pos[:, self.asset_cfg.joint_ids]
    if bool(self.needs.any()):
      self.q0[self.needs] = q[self.needs]
      self.needs[:] = False
    diff = (torch.abs(q - self.q0) - joint_tolerance).clamp_min(0.0)
    val = diff.square().sum(-1)
    return val / q.shape[1] if average_per_joint else val


# --------------------------------------------------------------------------
# metrics (logged per episode as Episode_Metrics/<name>)
# --------------------------------------------------------------------------
class rotation_progress_physical:
  """Original curriculum progress metric with physical tilt and fresh anchor."""

  def __init__(self, cfg, env):
    self.anchor = _CubeAnchor(env)

  def reset(self, env_ids=None):
    self.anchor.mark(env_ids)

  def __call__(self, env, target_yaw_rate: float = 0.2, position_threshold: float = 0.02, tilt_threshold: float = 0.35):
    anchored = self.anchor.update()
    w = self.anchor.cube.data.root_link_ang_vel_w[:, 2]
    yaw_score = (w / target_yaw_rate).clamp(0.0, 1.0)
    pos_score = (1 - self.anchor.drift() / position_threshold).clamp(0.0, 1.0)
    tilt_score = (1 - self.anchor.tilt() / tilt_threshold).clamp(0.0, 1.0)
    p = torch.nan_to_num(yaw_score * pos_score * tilt_score, nan=0.0)
    return torch.where(anchored, torch.zeros_like(p), p)


class cube_motion_metric:
  """component: 'yaw_rate' (genuine +Z rad/s), 'tilt' (rad), 'drift' (m)."""

  def __init__(self, cfg, env):
    self.anchor = _CubeAnchor(env)
    self.yaw = _YawTracker(env)

  def reset(self, env_ids=None):
    self.anchor.mark(env_ids)
    self.yaw.mark(env_ids)

  def __call__(self, env, component: str):
    self.anchor.update()
    d = self.yaw.step()
    if component == "yaw_rate":
      return d / env.step_dt
    if component == "tilt":
      return self.anchor.tilt()
    if component == "drift":
      return self.anchor.drift()
    raise ValueError(component)


def executed_action_saturation(env, action_name: str = "joint_pos", threshold: float = 0.99):
  """Fraction of executed normalised action entries with |a| > threshold."""
  term = env.action_manager.get_term(action_name)
  a = getattr(term, "executed_normalized", None)
  if a is None:
    a = term.raw_action.clamp(-1.0, 1.0)
  return (a.abs() > threshold).float().mean(-1)


# --------------------------------------------------------------------------
# observations (the "cube_state" group: oracle input for Experiment D and the
# label source for the estimator; never part of the A / C actor input)
# --------------------------------------------------------------------------
def fingertip_contacts(env, sensor_name: str = "fingertip_cube_contact"):
  found = env.scene[sensor_name].data.found
  return (found > 0).float()
