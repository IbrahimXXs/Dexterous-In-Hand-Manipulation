"""Grasp-Rotation-V2 reward terms (rewards only; nothing here reaches the actor).

``retention_penalty_v2`` is Retention-V1's ``retention_penalty`` with its two parts
weighted separately:

    p = pos_scale * sum_k clamp(excess_k / sd_k, 0, cap)^2      (risky +x, +y, -z)
      + tilt_scale_factor * clamp((tilt - tilt_margin) / 0.1, 0, cap)^2

With ``pos_scale = tilt_scale_factor = 1`` and ``tilt_margin = 0.5`` it is exactly the
Retention-V1 term (same statistics, margins, cap, anchor and palm frame); every V2
candidate changes only the documented parameters.
"""

from __future__ import annotations

from in_hand_rotation_mjlab.retention_v1.rewards import TILT_MARGIN_RAD, retention_excess
from in_hand_rotation_mjlab.scratch_study.mdp import _CubeAnchor


def retention_v2_raw(pos_palm, tilt, *, pos_scale: float = 1.0, tilt_scale_factor: float = 1.0,
                     tilt_margin: float = TILT_MARGIN_RAD):
  """Raw V2 retention penalty (>= 0); equals ``retention_excess`` at the defaults."""
  zero = tilt.new_zeros(tilt.shape)
  pos_part = retention_excess(pos_palm, zero)
  tilt_part = retention_excess(pos_palm, tilt, tilt_margin=tilt_margin) - retention_excess(pos_palm, zero, tilt_margin=tilt_margin)
  return pos_scale * pos_part + tilt_scale_factor * tilt_part


class retention_penalty_v2:
  def __init__(self, cfg, env):
    self.anchor = _CubeAnchor(env, cfg.params.get("object_name", "cube"))

  def reset(self, env_ids=None):
    self.anchor.mark(env_ids)

  def __call__(self, env, object_name: str = "cube", hand_cfg=None, palm_center_geom_expr: str = "palm_collision_.*",
               pos_scale: float = 1.0, tilt_scale_factor: float = 1.0, tilt_margin: float = TILT_MARGIN_RAD):
    from in_hand_rotation_mjlab.tasks.hand_cube.mdp.observations import cube_pose_in_palm_frame

    self.anchor.update()
    pos = cube_pose_in_palm_frame(env, object_name=object_name, hand_cfg=hand_cfg, palm_center_geom_expr=palm_center_geom_expr)[:, :3]
    return retention_v2_raw(pos, self.anchor.tilt(), pos_scale=pos_scale, tilt_scale_factor=tilt_scale_factor,
                            tilt_margin=tilt_margin)
