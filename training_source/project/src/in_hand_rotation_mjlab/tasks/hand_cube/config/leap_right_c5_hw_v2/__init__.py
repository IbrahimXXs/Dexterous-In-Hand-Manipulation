"""Task id of C5-HW-V2 (experiments/c5_hw_v2/PROTOCOL.md; single-seed exploratory).

``Mjlab-Leap-C5HW2``: C5-HW-EFFORT + the fitted actuator response + kit-start resets (``env_cfgs.py``);
the hardware-limited A PPO configuration (3001 updates, zero-initialised actor output, as C5 and
C5-HW-EFFORT), logged under ``logs/rsl_rl/c5_hw_v2``.  No existing task is modified.
"""

from mjlab.tasks.registry import register_mjlab_task

from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_hwlimit_study.rl_cfg import hwlimit_ppo_cfg

from .env_cfgs import KIT_START_CACHE, c5hw2_env_cfg


def _rl_cfg():
  cfg = hwlimit_ppo_cfg("A")
  cfg.experiment_name = "c5_hw_v2"
  return cfg


if KIT_START_CACHE.is_file():  # the cache is produced by experiments/c5_hw_v2/scripts/make_kit_start_cache.py
  register_mjlab_task(task_id="Mjlab-Leap-C5HW2", env_cfg=c5hw2_env_cfg(), play_env_cfg=c5hw2_env_cfg(play=True),
                      rl_cfg=_rl_cfg())
