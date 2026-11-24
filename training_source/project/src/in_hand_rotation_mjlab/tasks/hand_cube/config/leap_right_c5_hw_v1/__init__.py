"""Task id of C5-HW-V1 condition C5-HW-effort (experiments/c5_hw_v1/SPEC.md).

``Mjlab-Leap-C5HW-EFFORT``: GRV2 C5 with the deployed 350 mA effort limit and the actuator
utilisation penalty (``env_cfgs.py``); the hardware-limited A PPO configuration (3001 updates,
zero-initialised actor output, as C5), logged under ``logs/rsl_rl/c5_hw_v1_EFFORT``.
No existing task is modified.
"""

from mjlab.tasks.registry import register_mjlab_task

from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_hwlimit_study.rl_cfg import hwlimit_ppo_cfg

from .env_cfgs import EFFORT_PENALTY_WEIGHT, c5hw_effort_env_cfg


def _rl_cfg():
  cfg = hwlimit_ppo_cfg("A")
  cfg.experiment_name = "c5_hw_v1_EFFORT"
  return cfg


if EFFORT_PENALTY_WEIGHT is not None:
  register_mjlab_task(task_id="Mjlab-Leap-C5HW-EFFORT", env_cfg=c5hw_effort_env_cfg(),
                      play_env_cfg=c5hw_effort_env_cfg(play=True), rl_cfg=_rl_cfg())
