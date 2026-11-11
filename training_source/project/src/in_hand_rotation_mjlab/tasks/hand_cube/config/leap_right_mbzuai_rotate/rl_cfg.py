"""PPO settings for continuous MBZUAI cube rotation."""

from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_upstream_baseline.rl_cfg import (
  leap_right_upstream_baseline_ppo_cfg,
)
from mjlab.rl import RslRlOnPolicyRunnerCfg


def leap_right_mbzuai_rotate_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  """Retain the model_9999 architecture and use a lower fine-tuning rate."""
  cfg = leap_right_upstream_baseline_ppo_cfg()
  cfg.experiment_name = "leap_right_hand_cube_mbzuai_rotate"
  cfg.algorithm.learning_rate = 3e-4
  cfg.max_iterations = 3_000
  cfg.save_interval = 100
  cfg.logger = "tensorboard"
  return cfg
