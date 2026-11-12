"""Original MBZUAI continuous-yaw environment with the corrected reset anchor."""

from mjlab.envs import ManagerBasedRlEnvCfg

from in_hand_rotation_mjlab.estimation.env_fixes import apply_reset_anchor_fix
from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_mbzuai_rotate.env_cfgs import (
  leap_right_mbzuai_rotate_env_cfg,
)


def fixed_reset_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = leap_right_mbzuai_rotate_env_cfg(play=play)
  apply_reset_anchor_fix(cfg)
  return cfg
