"""PPO settings for Playground-style right-hand cube reorientation."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right.rl_cfg import (
  leap_right_hand_cube_rotate_ppo_cfg,
)


def leap_right_playground_reorient_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  """Match the public network width while retaining the working RSL-RL stack."""
  cfg = leap_right_hand_cube_rotate_ppo_cfg()
  cfg.experiment_name = "leap_right_hand_cube_playground_reorient"
  cfg.actor.hidden_dims = (512, 256, 128)
  cfg.critic.hidden_dims = (512, 256, 128)
  cfg.actor.init_noise_std = 0.7
  cfg.actor.noise_std_type = "scalar"
  cfg.algorithm.entropy_coef = 0.01
  cfg.algorithm.learning_rate = 3e-4
  cfg.num_steps_per_env = 40
  cfg.save_interval = 25
  cfg.max_iterations = 2_000
  cfg.logger = "tensorboard"
  return cfg


def leap_right_playground_reorient_tilted_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  """Use separate logs while retaining checkpoint-compatible PPO settings."""
  cfg = leap_right_playground_reorient_ppo_cfg()
  cfg.experiment_name = "leap_right_hand_cube_playground_reorient_tilted"
  return cfg


leap_right_playground_reorient_rl_cfg = (
  leap_right_playground_reorient_ppo_cfg
)
