"""PPO settings for contact-preserving palm lift and yaw."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_visual_yaw.rl_cfg import (
  leap_right_visual_yaw_ppo_cfg,
)


def leap_right_palm_lift_yaw_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = leap_right_visual_yaw_ppo_cfg()
  cfg.experiment_name = "leap_right_hand_cube_palm_lift_yaw"
  cfg.max_iterations = 10_000
  cfg.save_interval = 100
  return cfg


leap_right_palm_lift_yaw_rl_cfg = leap_right_palm_lift_yaw_ppo_cfg
