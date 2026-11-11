"""PPO configuration for hardware-aligned 60-second cube rotation."""

from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_upstream_baseline.rl_cfg import (
    leap_right_upstream_baseline_ppo_cfg,
)
from mjlab.rl import RslRlOnPolicyRunnerCfg


def leap_right_hardware_60s_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    """Reuse upstream PPO hyperparameters under a separate experiment."""
    cfg = leap_right_upstream_baseline_ppo_cfg()
    # Actor-only bootstrapping intentionally does not copy the old policy stochastic
    # standard deviation. Fresh 0.7-rad exploration immediately destroys an
    # already-skilled grasp, so resume conservatively and let PPO adapt from here.
    cfg.actor.noise_std_type = "log"
    cfg.actor.init_noise_std = 0.10
    cfg.experiment_name = "leap_right_hand_cube_rotate_hardware_60s"
    cfg.max_iterations = 10_000
    cfg.logger = "tensorboard"
    return cfg
