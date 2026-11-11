"""PPO configuration for arbitrary right-hand cube reorientation."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right.rl_cfg import (
  leap_right_hand_cube_rotate_ppo_cfg,
)


def leap_right_hand_cube_reorient_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  """Use conservative exploration for a contact-rich camera policy."""
  cfg = leap_right_hand_cube_rotate_ppo_cfg()
  cfg.experiment_name = "leap_right_hand_cube_reorient"
  cfg.actor.noise_std_type = "log"
  cfg.actor.init_noise_std = 0.30
  cfg.algorithm.entropy_coef = 0.0
  cfg.algorithm.learning_rate = 3e-4
  cfg.algorithm.clip_param = 0.20
  cfg.algorithm.desired_kl = 0.01
  cfg.max_iterations = 10_000
  cfg.save_interval = 25
  return cfg


leap_right_hand_cube_reorient_rl_cfg = leap_right_hand_cube_reorient_ppo_cfg

