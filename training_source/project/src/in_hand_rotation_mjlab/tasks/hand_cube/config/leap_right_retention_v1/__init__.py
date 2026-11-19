"""Task id of the Retention-V1 experiment (experiments/retention_v1).

``Mjlab-Leap-RetV1-A``: proprioception-only actor (320-D, as hardware-limited A), the
hardware-limited environment plus the Retention-V1 reward terms, and the
hardware-limited A PPO configuration (3001 updates, zero-initialised actor output),
logged under ``logs/rsl_rl/retention_v1_A``.  No existing task is modified.
"""

from mjlab.tasks.registry import register_mjlab_task

from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_hwlimit_study.rl_cfg import hwlimit_ppo_cfg

from .env_cfgs import retention_env_cfg


def _rl_cfg():
  cfg = hwlimit_ppo_cfg("A")
  cfg.experiment_name = "retention_v1_A"
  return cfg


register_mjlab_task(
  task_id="Mjlab-Leap-RetV1-A",
  env_cfg=retention_env_cfg(),
  play_env_cfg=retention_env_cfg(play=True),
  rl_cfg=_rl_cfg(),
)
