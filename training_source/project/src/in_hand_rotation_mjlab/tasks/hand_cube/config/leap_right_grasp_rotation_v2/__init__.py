"""Task ids of the Grasp-Rotation-V2 reward-balance pilot (experiments/grasp_rotation_v2).

``Mjlab-Leap-GRV2-P0`` and ``Mjlab-Leap-GRV2-C5``: Retention-V1 with one reward change each
(see ``env_cfgs.py``), the hardware-limited A PPO configuration (3001 updates,
zero-initialised actor output), logged under ``logs/rsl_rl/grasp_rotation_v2_<cand>``.
No existing task is modified.
"""

from mjlab.tasks.registry import register_mjlab_task

from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_hwlimit_study.rl_cfg import hwlimit_ppo_cfg

from .env_cfgs import CANDIDATES, grv2_env_cfg


def _rl_cfg(cand: str):
  cfg = hwlimit_ppo_cfg("A")
  cfg.experiment_name = f"grasp_rotation_v2_{cand}"
  return cfg


for _c in CANDIDATES:
  register_mjlab_task(task_id=f"Mjlab-Leap-GRV2-{_c}", env_cfg=grv2_env_cfg(_c), play_env_cfg=grv2_env_cfg(_c, play=True),
                      rl_cfg=_rl_cfg(_c))
