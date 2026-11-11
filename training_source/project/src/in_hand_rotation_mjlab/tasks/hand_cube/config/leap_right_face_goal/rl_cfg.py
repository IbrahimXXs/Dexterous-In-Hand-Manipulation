from mjlab.rl import RslRlOnPolicyRunnerCfg

from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right.rl_cfg import (
  leap_right_hand_cube_rotate_ppo_cfg,
)


def leap_right_hand_cube_face_up_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  """Conservative PPO settings for camera-conditioned face reorientation."""
  cfg = leap_right_hand_cube_rotate_ppo_cfg()
  cfg.experiment_name = "leap_right_hand_cube_face_up"
  # Keep log-space positivity, but remove the unbounded entropy incentive that
  # drove the old log_std from about 0 to 12 and saturated every joint action.
  cfg.actor.noise_std_type = "log"
  cfg.actor.init_noise_std = 0.30
  cfg.algorithm.entropy_coef = 0.0
  cfg.algorithm.learning_rate = 3e-4
  cfg.algorithm.clip_param = 0.20
  cfg.algorithm.desired_kl = 0.01
  cfg.max_iterations = 10_000
  cfg.save_interval = 25
  return cfg


leap_right_hand_cube_face_up_rl_cfg = leap_right_hand_cube_face_up_ppo_cfg
