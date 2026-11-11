"""PPO settings for flat camera-guided right-hand cube yaw control."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right.rl_cfg import (
  leap_right_hand_cube_rotate_ppo_cfg,
)


def leap_right_visual_yaw_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  """Match the public network width while retaining the working RSL-RL stack."""
  cfg = leap_right_hand_cube_rotate_ppo_cfg()
  cfg.experiment_name = "leap_right_hand_cube_mbzuai_visual_yaw"
  cfg.actor.hidden_dims = (512, 512, 256)
  cfg.critic.hidden_dims = (512, 512, 256)
  cfg.actor.init_noise_std = 0.7
  cfg.actor.noise_std_type = "scalar"
  cfg.algorithm.entropy_coef = 0.003
  cfg.algorithm.learning_rate = 1e-3
  cfg.num_steps_per_env = 32
  cfg.save_interval = 100
  cfg.max_iterations = 5_000
  cfg.logger = "tensorboard"
  return cfg


leap_right_visual_yaw_rl_cfg = (
  leap_right_visual_yaw_ppo_cfg
)
