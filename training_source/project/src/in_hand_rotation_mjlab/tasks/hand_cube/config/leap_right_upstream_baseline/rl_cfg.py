"""PPO configuration for the longer upstream-reward right-hand baseline."""

from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right.rl_cfg import (
  leap_right_hand_cube_rotate_ppo_cfg,
)
from mjlab.rl import RslRlOnPolicyRunnerCfg


def leap_right_upstream_baseline_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  """Use upstream PPO hyperparameters, extended from 5k to 10k iterations."""
  cfg = leap_right_hand_cube_rotate_ppo_cfg()
  cfg.experiment_name = "leap_right_hand_cube_rotate_upstream_baseline"
  cfg.max_iterations = 10_000
  cfg.logger = "tensorboard"
  return cfg
