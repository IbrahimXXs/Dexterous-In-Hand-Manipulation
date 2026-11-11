"""PPO configurations for autonomous cylinder pickup and translation."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_grasp.rl_cfg import (
  leap_right_hand_cube_grasp_ppo_cfg,
)


def leap_right_hand_cylinder_pickup_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  """Fine-tune the same-hand table-grasp actor for cylinder geometry."""
  cfg = leap_right_hand_cube_grasp_ppo_cfg()
  cfg.experiment_name = "leap_right_hand_cylinder_pickup"
  # Delta actions are integrated into joint targets; 0.10 accumulates enough
  # random walk to destroy a demonstrated grasp before PPO sees its first
  # completion reward.
  cfg.actor.init_noise_std = 0.02
  cfg.algorithm.entropy_coef = 0.0005
  cfg.algorithm.learning_rate = 5e-5
  cfg.num_steps_per_env = 32
  cfg.save_interval = 50
  cfg.max_iterations = 1_500
  return cfg


def leap_right_hand_cylinder_pickup_translate_ppo_cfg() -> (
  RslRlOnPolicyRunnerCfg
):
  """Extend the accepted pickup actor with a palm-X translation goal."""
  cfg = leap_right_hand_cylinder_pickup_ppo_cfg()
  cfg.experiment_name = "leap_right_hand_cylinder_pickup_translate"
  cfg.actor.init_noise_std = 0.05
  cfg.algorithm.entropy_coef = 0.0002
  cfg.algorithm.learning_rate = 3e-5
  cfg.save_interval = 50
  cfg.max_iterations = 1_500
  return cfg
