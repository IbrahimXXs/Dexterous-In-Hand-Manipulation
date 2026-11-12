"""Corrected environment for the from-scratch state-estimation study.

Built from the original ``Mjlab-Leap-Right-HandCube-MBZUAI-Rotate`` config
(same robot, cube, grasp cache, domain randomisation, reward weights,
thresholds, curriculum, observation noise/delay, 20 s episodes).  Changes:

1. ``cube_pose_deviation``: fresh post-reset anchor + physical tilt
   (thresholds unchanged: 0.08 m, 0.8 rad).
2. ``rotate_finite_diff``: genuine (rotation-vector) yaw increment, physical
   tilt in the drift gate, fresh anchor (clip / gate values unchanged).
3. ``pose_diff_penalty``: fresh post-reset anchor (weights unchanged).
4. Metrics ``rotation_progress`` (drives the penalty curriculum),
   ``position_error``, ``tilt_error``: physical tilt, fresh anchor; new
   ``genuine_yaw_rate`` and ``exec_action_saturation`` metrics.
5. Action: identical delta-position mapping (+-1/24 rad per step), with the
   policy output squashed by tanh instead of clipped ("clip" kept for the
   pilot comparison).
6. A ``cube_state`` observation group (palm-frame cube pose, velocities,
   fingertip contacts).  It is read only by the Experiment-D actor and by the
   data collector; the A / C actors and every critic never see it.
"""

from __future__ import annotations

from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from in_hand_rotation_mjlab.scratch_study import mdp as S
from in_hand_rotation_mjlab.tasks.hand_cube import mdp as hand_cube_mdp
from .actions import SquashedJointPositionDeltaActionCfg
from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_mbzuai_rotate.env_cfgs import (
  leap_right_mbzuai_rotate_env_cfg,
)

_PALM = {
  "object_name": "cube",
  "hand_cfg": SceneEntityCfg("robot", body_names=("palm",)),
  "palm_center_geom_expr": "palm_collision_.*",
}


def scratch_env_cfg(play: bool = False, squash: str = "tanh"):
  cfg = leap_right_mbzuai_rotate_env_cfg(play=play)

  # 1. termination
  old = cfg.terminations["cube_pose_deviation"].params
  cfg.terminations["cube_pose_deviation"] = TerminationTermCfg(
    func=S.cube_pose_deviation_physical,
    params={"max_position_error": old["max_position_error"], "max_tilt_error": old["max_tilt_error"]},
  )

  # 2. rotation reward
  r = cfg.rewards["rotate_finite_diff"]
  p = dict(r.params)
  assert p.pop("negate_yaw_rate") is False and p.pop("drift_mode") == "step"
  cfg.rewards["rotate_finite_diff"] = RewardTermCfg(func=S.genuine_yaw_rate_reward, weight=r.weight, params=p)

  # 3. joint-pose penalty
  r = cfg.rewards["pose_diff_penalty"]
  cfg.rewards["pose_diff_penalty"] = RewardTermCfg(
    func=S.pose_diff_l2_from_reset_anchored, weight=r.weight, params=dict(r.params)
  )

  # 4. metrics
  mp = cfg.metrics["rotation_progress"].params
  cfg.metrics["rotation_progress"] = MetricsTermCfg(
    func=S.rotation_progress_physical,
    params={k: mp[k] for k in ("target_yaw_rate", "position_threshold", "tilt_threshold")},
  )
  cfg.metrics["position_error"] = MetricsTermCfg(func=S.cube_motion_metric, params={"component": "drift"})
  cfg.metrics["tilt_error"] = MetricsTermCfg(func=S.cube_motion_metric, params={"component": "tilt"})
  cfg.metrics["genuine_yaw_rate"] = MetricsTermCfg(func=S.cube_motion_metric, params={"component": "yaw_rate"})
  cfg.metrics["exec_action_saturation"] = MetricsTermCfg(func=S.executed_action_saturation, params={})

  # 5. action
  a = cfg.actions["joint_pos"]
  cfg.actions["joint_pos"] = SquashedJointPositionDeltaActionCfg(
    entity_name=a.entity_name,
    actuator_names=a.actuator_names,
    scale=a.scale,
    offset=a.offset,
    use_default_offset=a.use_default_offset,
    use_cached_reset_target=a.use_cached_reset_target,
    clip_to_joint_limits=a.clip_to_joint_limits,
    use_soft_joint_pos_limits=a.use_soft_joint_pos_limits,
    delta_min=a.delta_min,
    delta_max=a.delta_max,
    interpolate_decimation=a.interpolate_decimation,
    squash=squash,
  )

  # 6. ground-truth cube state (oracle input for D, labels for the estimator)
  cfg.observations["cube_state"] = ObservationGroupCfg(
    {
      "cube_pose_palm": ObservationTermCfg(func=hand_cube_mdp.cube_pose_in_palm_frame, params=dict(_PALM)),
      "cube_lin_vel_palm": ObservationTermCfg(func=hand_cube_mdp.cube_lin_vel_in_palm_frame, params=dict(_PALM)),
      "cube_ang_vel_palm": ObservationTermCfg(func=hand_cube_mdp.cube_ang_vel_in_palm_frame, params=dict(_PALM)),
      "fingertip_contacts": ObservationTermCfg(func=S.fingertip_contacts, params={}),
    },
    enable_corruption=False,
    history_length=1,
    flatten_history_dim=True,
  )
  return cfg
