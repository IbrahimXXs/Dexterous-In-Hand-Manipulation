"""Environment of the hardware-limit A/C study (experiments/hwlimit_ac).

Identical to the corrected from-scratch study environment
(``leap_right_scratch_study.env_cfgs.scratch_env_cfg``: robot, cube, rewards,
terminations, curriculum, observations, observation noise/delay, domain
randomisation, tanh-squashed delta-position action of +-1/24 rad per step,
20 Hz control), with exactly two changes, applied identically to A and C:

1. Joint-command limits.  Every joint-position command (the integrated
   target, the quantity the deployment kit sends to the motors) is clamped to
   the deployment kit's approved limits (``hwlimit_study.limits``: kit limits
   with its 0.01 rad margin) intersected with the model's soft joint limits,
   instead of the model's soft limits alone.  The clamped command is what the
   motors track and what the ``prev_commanded_joint_pos`` observation reports,
   exactly as on the robot.  MuJoCo's mechanical joint ranges are unchanged
   (the real joints can still be pushed past a command limit by contact).
2. Reset states.  Episodes start from a grasp cache whose every joint state
   and command lies inside those limits (``experiments/hwlimit_ac/cache/``,
   built by ``experiments/hwlimit_ac/scripts/build_grasp_cache.py`` from the
   original cache: projection into the limits, settling, and the original
   validator's acceptance criteria).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import torch

from in_hand_rotation_mjlab.hwlimit_study.limits import approved_command_limits
from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_scratch_study.actions import (
  SquashedJointPositionDeltaAction,
  SquashedJointPositionDeltaActionCfg,
)
from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_scratch_study.env_cfgs import scratch_env_cfg

_PROJECT = Path(__file__).resolve().parents[6]
EXPERIMENT_DIR = _PROJECT / "experiments" / "hwlimit_ac"
HWLIMIT_GRASP_CACHE = EXPERIMENT_DIR / "cache" / "hwlimit_grasp_cache.npz"


@dataclass(kw_only=True)
class HwLimitedJointPositionDeltaActionCfg(SquashedJointPositionDeltaActionCfg):
  """Squashed delta-position action whose command is clamped to the approved limits."""

  limit_margin_rad: float = 0.01
  required_grasp_cache: str | None = None  # refuse to run on the silent default-reset fallback

  def build(self, env):
    return HwLimitedJointPositionDeltaAction(self, env)


class HwLimitedJointPositionDeltaAction(SquashedJointPositionDeltaAction):
  cfg: HwLimitedJointPositionDeltaActionCfg

  def __init__(self, cfg: HwLimitedJointPositionDeltaActionCfg, env):
    if cfg.required_grasp_cache is not None and not Path(cfg.required_grasp_cache).is_file():
      raise FileNotFoundError(f"hardware-limit grasp cache missing: {cfg.required_grasp_cache}")
    if not (cfg.clip_to_joint_limits and cfg.use_soft_joint_pos_limits):
      raise ValueError("expected the study action to clip to soft joint limits")
    self._approved: tuple[torch.Tensor, torch.Tensor] | None = None
    super().__init__(cfg, env)

  def approved_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
    if self._approved is None:
      soft = self._entity.data.soft_joint_pos_limits[0, self._target_ids].detach().cpu().numpy()
      lo, hi = approved_command_limits(soft[:, 0], soft[:, 1], self.cfg.limit_margin_rad)
      dev, dt = self._raw_actions.device, self._raw_actions.dtype
      self._approved = (torch.as_tensor(lo, device=dev, dtype=dt), torch.as_tensor(hi, device=dev, dtype=dt))
    return self._approved

  def _joint_limits(self, env_ids):
    lo, hi = self.approved_limits()
    n = self._target[env_ids].shape[0]
    return lo.expand(n, -1), hi.expand(n, -1)


def hwlimit_env_cfg(play: bool = False, grasp_cache: str | Path = HWLIMIT_GRASP_CACHE):
  cfg = scratch_env_cfg(play=play, squash="tanh")
  a = cfg.actions["joint_pos"]
  kw = {f.name: getattr(a, f.name) for f in fields(SquashedJointPositionDeltaActionCfg) if f.init}
  cfg.actions["joint_pos"] = HwLimitedJointPositionDeltaActionCfg(**kw, required_grasp_cache=str(grasp_cache))
  cfg.events["reset_from_grasp_cache"].params["cache_file"] = str(grasp_cache)
  return cfg
