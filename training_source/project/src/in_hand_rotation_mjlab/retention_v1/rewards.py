"""Retention-V1 reward terms (rewards only; nothing here reaches the actor).

Derived from the Phase-1 diagnosis (experiments/retention_v1/README.md Section 2;
results/diagnosis.json), on 3072 60-s rollouts of the three hardware-limited A policies:

* 99.4% of failures are pose failures by TILT (> 0.8 rad from the episode-initial
  attitude); drift failures are 0.6%.  Pose-deviation terminations carry no explicit
  penalty in the current reward (only ``object_fallen`` = cube below 0.2 m does).
* The 2-s failure hazard rises one-sidedly with the cube-centre position in the palm
  frame (survivor mean mu, s.d. sd below): beyond +2 sd in x 14.5x the base hazard,
  beyond +2 sd in y 5x (beyond +3 sd 18.5x), beyond -2 sd in z 6.7x (beyond -3 sd 18.8x);
  the opposite directions stay at 0.5-2.4x.  It rises with tilt above 0.5 rad
  (0.5-0.6: 1.9x, 0.6-0.7: 6.7x, 0.7-0.8: 25x).  Position excursions precede 97% of
  pose failures (median lead 6 s), tilt growth the last 2-3 s.

Terms
-----
``retention_penalty`` (raw value >= 0, used with a negative weight):

    p = sum over the risky directions (+x, +y, -z) of clamp(excess_k / sd_k, 0, 3)^2
      + clamp((tilt - 0.5) / 0.1, 0, 3)^2

    excess_k = signed palm-frame distance beyond mu_k + 1.5 sd_k (direction-aware).
  Zero inside the useful region (all survivors' typical positions and tilt <= 0.5 rad,
  i.e. no change to normal rotation or finger repositioning); quadratic outside;
  each component capped at 9 (3 s.d. / 0.3 rad beyond the margin).

``grasp_loss`` (raw value 0/1, used with a negative weight): 1 on the step where the
existing ``cube_pose_deviation`` termination fires and ``cube_fell`` does not (the latter
already costs ``object_fallen``), so it is paid once per genuine grasp loss and never on
a timeout.
"""

from __future__ import annotations

import torch

from in_hand_rotation_mjlab.scratch_study.mdp import _CubeAnchor

# Survivor palm-frame statistics of the hwlimit A policies (diagnosis.json, pooled, m)
PALM_MU = (0.0417, 0.0022, -0.0543)
PALM_SD = (0.0099, 0.0099, 0.0032)
RISK_SIGN = (1.0, 1.0, -1.0)  # risky directions: +x, +y, -z
MARGIN_SD = 1.5
TILT_MARGIN_RAD = 0.5
TILT_SCALE_RAD = 0.1
CAP = 3.0


def retention_excess(pos_palm: torch.Tensor, tilt: torch.Tensor, *, margin_sd: float = MARGIN_SD,
                     tilt_margin: float = TILT_MARGIN_RAD, tilt_scale: float = TILT_SCALE_RAD, cap: float = CAP) -> torch.Tensor:
  """Raw retention penalty (>= 0) for cube centre ``pos_palm`` [N,3] (m) and tilt [N] (rad)."""
  mu = torch.tensor(PALM_MU, device=pos_palm.device, dtype=pos_palm.dtype)
  sd = torch.tensor(PALM_SD, device=pos_palm.device, dtype=pos_palm.dtype)
  sign = torch.tensor(RISK_SIGN, device=pos_palm.device, dtype=pos_palm.dtype)
  z = sign * (pos_palm - mu) / sd  # positive = towards the risky side, in s.d.
  pos_term = (z - margin_sd).clamp(0.0, cap).square().sum(-1)
  tilt_term = ((tilt - tilt_margin) / tilt_scale).clamp(0.0, cap).square()
  return pos_term + tilt_term


class retention_penalty:
  """Stateful wrapper: palm-frame cube position + tilt from the episode-initial attitude
  (anchored on the first fresh post-reset step, as the study's termination and rewards)."""

  def __init__(self, cfg, env):
    self.anchor = _CubeAnchor(env, cfg.params.get("object_name", "cube"))

  def reset(self, env_ids=None):
    self.anchor.mark(env_ids)

  def __call__(self, env, object_name: str = "cube", hand_cfg=None, palm_center_geom_expr: str = "palm_collision_.*"):
    from in_hand_rotation_mjlab.tasks.hand_cube.mdp.observations import cube_pose_in_palm_frame

    self.anchor.update()
    pos = cube_pose_in_palm_frame(env, object_name=object_name, hand_cfg=hand_cfg, palm_center_geom_expr=palm_center_geom_expr)[:, :3]
    return retention_excess(pos, self.anchor.tilt())


def grasp_loss(env, pose_term: str = "cube_pose_deviation", fell_term: str = "cube_fell") -> torch.Tensor:
  """1.0 on the step a pose-deviation termination fires (and the cube did not fall), else 0."""
  tm = env.termination_manager
  pose = tm.get_term(pose_term).bool()
  fell = tm.get_term(fell_term).bool() if fell_term in tm.active_terms else torch.zeros_like(pose)
  return (pose & ~fell).float()
