"""Grasp-Rotation-V2 reward-balance candidates (experiments/grasp_rotation_v2, PROTOCOL.md Section 2).

Each candidate is the Retention-V1 environment (``retention_env_cfg``: hardware-limited
environment, original 239-grasp reset distribution, observations, actions, limits, physics,
domain randomisation, curriculum, terminations and the other reward terms unchanged) with
exactly ONE documented reward change:

* ``P0`` (hypothesis H1): retention penalty without its palm-position part
  (``pos_scale = 0``); its tilt part (margin 0.5 rad) and ``grasp_loss`` (-10) are kept.
* ``C5`` (hypothesis H4): yaw-rate reward clip raised from +-0.25 to +-0.5 rad/s; the
  Retention-V1 retention penalty and ``grasp_loss`` are kept unchanged.
"""

from __future__ import annotations

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from in_hand_rotation_mjlab.grasp_rotation_v2.rewards import retention_penalty_v2
from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_retention_v1.env_cfgs import RETENTION_WEIGHT, retention_env_cfg

CANDIDATES = {
  "P0": {"retention": {"pos_scale": 0.0, "tilt_scale_factor": 1.0, "tilt_margin": 0.5}, "yaw_clip": 0.25},
  "C5": {"retention": None, "yaw_clip": 0.5},
}


def grv2_env_cfg(candidate: str, play: bool = False):
  spec = CANDIDATES[candidate]
  cfg = retention_env_cfg(play=play)
  if spec["retention"] is not None:
    cfg.rewards["retention_penalty"] = RewardTermCfg(
      func=retention_penalty_v2, weight=RETENTION_WEIGHT,
      params={"object_name": "cube", "hand_cfg": SceneEntityCfg("robot", body_names=("palm",)),
              "palm_center_geom_expr": "palm_collision_.*", **spec["retention"]},
    )
  c = float(spec["yaw_clip"])
  cfg.rewards["rotate_finite_diff"].params["clip_min"] = -c
  cfg.rewards["rotate_finite_diff"].params["clip_max"] = c
  return cfg
