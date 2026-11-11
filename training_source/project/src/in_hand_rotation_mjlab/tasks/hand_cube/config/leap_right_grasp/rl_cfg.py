from mjlab.rl import RslRlOnPolicyRunnerCfg

from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right.rl_cfg import (
  leap_right_hand_cube_rotate_ppo_cfg,
)


def leap_right_hand_cube_grasp_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  """Canonical right-hand PPO model under a grasp-specific log namespace."""
  cfg = leap_right_hand_cube_rotate_ppo_cfg()
  cfg.experiment_name = "leap_right_hand_cube_grasp"
  cfg.actor.noise_std_type = "log"
  cfg.max_iterations = 10_000
  cfg.save_interval = 100
  return cfg


leap_right_hand_cube_grasp_rl_cfg = leap_right_hand_cube_grasp_ppo_cfg
