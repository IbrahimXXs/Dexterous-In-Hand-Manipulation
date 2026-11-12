"""Opt-in correction for the reset anchor of ``cube_pose_deviation``.

Finding (verified in ``tests/test_estimation_env.py::test_reset_anchor``)
-------------------------------------------------------------------------
``tasks/hand_cube/mdp/terminations.py::object_pose_rp_position_deviation_from_reset``
records the "episode-initial" cube pose inside ``reset()``.  MJLab v1.1.1 calls
manager ``reset()`` from ``_reset_idx`` *before* ``sim.forward()``, so
``root_link_pos_w`` / ``root_link_quat_w`` still hold the previous episode's
final pose (MJLab documents this staleness in ``ManagerBasedRlEnv.step``).
Because the grasp cache restores the cube in many different orientations, the
first post-reset step usually sees a roll/pitch "deviation" above 0.8 rad and
terminates again, cascading until a stale pose happens to match.

The yaw reward in the same task already avoids this by re-anchoring on the
first post-reset step (``episode_length_buf <= 1``).  This module applies the
same rule to the termination, with identical thresholds.  The original task
and checkpoint are not modified; the fix is enabled per-run via
``make_env(..., fix_reset_anchor=True)`` or the registered ``-FixedReset``
task variants in ``estimation/tasks.py``.
"""

from __future__ import annotations

import torch

from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat, wrap_to_pi

from in_hand_rotation_mjlab.estimation.rotations import quat_to_matrix
from in_hand_rotation_mjlab.tasks.hand_cube.mdp.terminations import (
  object_pose_rp_position_deviation_from_reset,
)


class object_pose_rp_position_deviation_from_reset_anchored(
  object_pose_rp_position_deviation_from_reset
):
  """Same check, anchored on the first post-reset step (fresh kinematics)."""

  def __call__(self, env, max_position_error, max_tilt_error, asset_cfg=None):
    pos_w = self._asset.data.root_link_pos_w
    roll, pitch, _ = euler_xyz_from_quat(self._asset.data.root_link_quat_w)
    anchor = env.episode_length_buf <= 1
    if torch.any(anchor):
      self._init_pos_w[anchor] = pos_w[anchor]
      self._init_roll[anchor] = roll[anchor]
      self._init_pitch[anchor] = pitch[anchor]
      self._has_init[anchor] = True
    pos_error = torch.linalg.vector_norm(pos_w - self._init_pos_w, dim=-1)
    tilt_error = torch.linalg.vector_norm(
      torch.stack(
        [
          wrap_to_pi(roll - self._init_roll).abs(),
          wrap_to_pi(pitch - self._init_pitch).abs(),
        ],
        dim=-1,
      ),
      dim=-1,
    )
    return self._has_init & ~anchor & (
      (pos_error > max_position_error) | (tilt_error > max_tilt_error)
    )


def apply_reset_anchor_fix(env_cfg) -> None:
  """Swap the termination function in-place on an env config."""
  term: TerminationTermCfg = env_cfg.terminations["cube_pose_deviation"]
  if term.func is not object_pose_rp_position_deviation_from_reset:
    raise RuntimeError(
      f"Unexpected cube_pose_deviation func {term.func!r}; refusing to patch."
    )
  term.func = object_pose_rp_position_deviation_from_reset_anchored


class object_pose_physical_tilt_position_deviation_anchored(
  object_pose_rp_position_deviation_from_reset
):
  """Anchored check whose tilt is the *physical* tilt angle.

  Second finding (see docs/state_estimation.md, diagnostics): the original
  tilt error is the norm of XYZ-Euler roll/pitch differences.  Near
  |pitch| = 90 deg (gimbal lock), which the grasp cache's cube orientations
  often hit, a few degrees of real tilt change roll/pitch by tens of degrees,
  so those episodes terminate almost surely.  Here tilt = angle between the
  world z-axis expressed in the cube frame now and at the anchor step, which
  is independent of yaw, of the Euler convention and of the cube's starting
  orientation.  Thresholds are unchanged.
  """

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self._init_up = torch.zeros((env.num_envs, 3), device=env.device)

  def _up_in_cube(self) -> torch.Tensor:
    return quat_to_matrix(self._asset.data.root_link_quat_w)[..., 2, :]

  def __call__(self, env, max_position_error, max_tilt_error, asset_cfg=None):
    pos_w = self._asset.data.root_link_pos_w
    up = self._up_in_cube()
    anchor = env.episode_length_buf <= 1
    if torch.any(anchor):
      self._init_pos_w[anchor] = pos_w[anchor]
      self._init_up[anchor] = up[anchor]
      self._has_init[anchor] = True
    pos_error = torch.linalg.vector_norm(pos_w - self._init_pos_w, dim=-1)
    tilt = torch.acos((up * self._init_up).sum(-1).clamp(-1.0, 1.0))
    return self._has_init & ~anchor & ((pos_error > max_position_error) | (tilt > max_tilt_error))


def apply_termination_variant(env_cfg, variant: str) -> None:
  """'original' | 'fixed' (anchor) | 'physical_tilt' (anchor + physical tilt)
  | 'no_pose_termination' (cube_pose_deviation removed; drops still end)."""
  if variant == "original":
    return
  apply_reset_anchor_fix(env_cfg)
  if variant == "fixed":
    return
  if variant == "physical_tilt":
    env_cfg.terminations["cube_pose_deviation"].func = object_pose_physical_tilt_position_deviation_anchored
  elif variant == "no_pose_termination":
    del env_cfg.terminations["cube_pose_deviation"]
  else:
    raise ValueError(f"unknown termination variant {variant!r}")
