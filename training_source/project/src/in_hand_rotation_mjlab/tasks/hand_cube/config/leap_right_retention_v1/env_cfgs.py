"""Retention-V1 environment (experiments/retention_v1).

Exactly the hardware-limited study environment (``leap_right_hwlimit_study``: approved
joint-command limits with the kit's margin, feasible 239-grasp reset cache, 20-s
episodes, observations, physics, domain randomisation, rewards, curriculum and
terminations unchanged) plus two reward terms that use simulator state only:

* ``retention_penalty`` (weight -0.05): one-sided palm-frame workspace + tilt-margin
  penalty, zero inside the useful region (``retention_v1/rewards.py``);
* ``grasp_loss`` (weight -10.0): once on a pose-deviation termination, the same
  magnitude as the existing ``object_fallen`` (-10), not on timeouts or falls.

The actor observation (320-D proprioception), action, limits and PPO settings are not changed.
"""

from __future__ import annotations

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from in_hand_rotation_mjlab.retention_v1 import rewards as R
from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_hwlimit_study.env_cfgs import hwlimit_env_cfg

RETENTION_WEIGHT = -0.05
GRASP_LOSS_WEIGHT = -10.0


def retention_env_cfg(play: bool = False):
  cfg = hwlimit_env_cfg(play=play)
  cfg.rewards["retention_penalty"] = RewardTermCfg(
    func=R.retention_penalty, weight=RETENTION_WEIGHT,
    params={"object_name": "cube", "hand_cfg": SceneEntityCfg("robot", body_names=("palm",)),
            "palm_center_geom_expr": "palm_collision_.*"},
  )
  cfg.rewards["grasp_loss"] = RewardTermCfg(func=R.grasp_loss, weight=GRASP_LOSS_WEIGHT, params={})
  return cfg
