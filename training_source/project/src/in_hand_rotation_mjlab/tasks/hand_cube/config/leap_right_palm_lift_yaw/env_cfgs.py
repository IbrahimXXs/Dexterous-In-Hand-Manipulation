"""Two/three-finger palm lift, disturbance hold, and discrete yaw task."""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from in_hand_rotation_mjlab import MYMJLAB_SRC_PATH
from in_hand_rotation_mjlab.tasks.hand_cube import mdp as hand_cube_mdp
from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_visual_yaw.env_cfgs import (
  _MBZUAI_GRASP_CACHE,
  _joint_cfg,
  _palm_params,
  leap_right_visual_yaw_env_cfg,
)


TARGET_LIFT_M = 0.030
MINIMUM_CONTACTS = 2


def leap_right_palm_lift_yaw_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build a hardware-aligned pinch/lift/jiggle/yaw task.

  A reset draws a known feasible two- or three-tip grasp, restores its matched
  LEAP joint state, then applies only small palm-frame position/yaw variation.
  Completion requires a 30 mm lift from that episode's initial palm height,
  two or more active fingertip contacts, and a 90 degree yaw target.
  """
  cfg = leap_right_visual_yaw_env_cfg(play=play)

  cfg.commands["pose_goal"] = hand_cube_mdp.DiscreteYawPoseGoalCommandCfg(
    entity_name="cube",
    resampling_time_range=(1e9, 1e9),
    success_threshold=math.radians(15.0),
    success_dwell_steps=1,
    resample_on_success=False,
    target_yaw_choices=(-0.5 * math.pi, 0.5 * math.pi),
    debug_vis=play,
  )

  # Starts remain physically valid cache entries. These small offsets provide
  # meaningful palm-location diversity without turning reset into a drop task.
  grasp_reset = cfg.events["reset_from_grasp_cache"].params
  grasp_reset["cache_file"] = str(
    MYMJLAB_SRC_PATH / "tasks/hand_cube/cache" / _MBZUAI_GRASP_CACHE
  )
  grasp_reset["pose_range"] = {
    "x": (0.0, 0.0) if play else (-0.006, 0.006),
    "y": (0.0, 0.0) if play else (-0.006, 0.006),
    "z": (0.0, 0.0) if play else (-0.003, 0.003),
    "yaw": (0.0, 0.0) if play else (-math.radians(15.0), math.radians(15.0)),
  }
  grasp_reset["scale_list"] = None
  grasp_reset["scale_jitter"] = 0.0
  grasp_reset["restore_robot_joint_state"] = True
  cfg.actions["joint_pos"].use_cached_reset_target = True

  fingertip_cfg = SceneEntityCfg(
    "robot",
    geom_names=("if_tip", "mf_tip", "rf_tip", "th_tip"),
    preserve_order=True,
  )
  cube_cfg = SceneEntityCfg("cube", geom_names=("cube_geom",), preserve_order=True)

  cfg.rewards = {
    # The dense contact terms make a thumb+one/two-finger pinch viable without
    # demanding all four fingertips at every control step.
    "pinch_contact": RewardTermCfg(
      func=hand_cube_mdp.fingertip_cube_contact_fraction,
      weight=1.8,
      params={"sensor_name": "fingertip_cube_contact", "max_contacts": 3},
    ),
    "contact_participation": RewardTermCfg(
      func=hand_cube_mdp.fingertip_contact_participation,
      weight=0.25,
      params={
        "sensor_name": "fingertip_cube_contact",
        "window_steps": 20,
        "expected_fingertips": 4,
        "participation_power": 1.5,
      },
    ),
    "fingertip_proximity": RewardTermCfg(
      func=hand_cube_mdp.fingertip_cube_proximity,
      weight=0.20,
      params={
        "hand_cfg": fingertip_cfg,
        "cube_cfg": cube_cfg,
        "surface_margin": 0.015,
        "distance_scale": 0.030,
        "worst_tip_weight": 0.40,
      },
    ),
    # Lift is measured relative to each environment's post-reset palm height,
    # so randomized cached starts cannot earn lift credit by themselves.
    "palm_lift": RewardTermCfg(
      func=hand_cube_mdp.contact_gated_palm_lift_from_reset,
      weight=5.0,
      params={
        "object_name": "cube",
        "sensor_name": "fingertip_cube_contact",
        "target_lift_m": TARGET_LIFT_M,
        "minimum_contacts": MINIMUM_CONTACTS,
        **_palm_params(),
      },
    ),
    "palm_retention": RewardTermCfg(
      func=hand_cube_mdp.cube_palm_retention_reward,
      weight=1.0,
      params={"object_name": "cube", "safe_distance": 0.15, **_palm_params()},
    ),
    "yaw_alignment": RewardTermCfg(
      func=hand_cube_mdp.playground_orientation_reward,
      weight=3.0,
      params={"command_name": "pose_goal", "object_name": "cube"},
    ),
    # This bonus cannot be obtained by spinning or lifting after contact loss.
    "lifted_yaw_success": RewardTermCfg(
      func=hand_cube_mdp.contact_gated_palm_lift_yaw_success,
      weight=250.0,
      params={
        "object_name": "cube",
        "sensor_name": "fingertip_cube_contact",
        "command_name": "pose_goal",
        "target_lift_m": TARGET_LIFT_M,
        "yaw_threshold": math.radians(15.0),
        "minimum_contacts": MINIMUM_CONTACTS,
        **_palm_params(),
      },
    ),
    "action_rate": RewardTermCfg(
      func=hand_cube_mdp.playground_action_rate_cost,
      weight=-0.001,
    ),
    "energy": RewardTermCfg(
      func=hand_cube_mdp.playground_energy_cost,
      weight=-0.001,
      params={"asset_cfg": _joint_cfg()},
    ),
  }

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "cube_fell": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": 0.20, "asset_cfg": SceneEntityCfg("cube")},
    ),
    "cube_lost": TerminationTermCfg(
      func=hand_cube_mdp.object_palm_distance_above,
      params={"max_distance": 0.19, "object_name": "cube", **_palm_params()},
    ),
    "nan": TerminationTermCfg(func=envs_mdp.nan_detection),
  }

  # Small intermittent wrenches represent jiggle/handling disturbances. They
  # are omitted in play so that a hardware operator controls the disturbance.
  if not play:
    cfg.events["jiggle_cube"] = EventTermCfg(
      func=hand_cube_mdp.apply_random_cube_wrench,
      mode="interval",
      interval_range_s=(4.0, 7.0),
      params={
        "asset_cfg": SceneEntityCfg("cube", body_names=("cube",)),
        "force_abs_max": 0.20,
        "torque_abs_max": 0.006,
        "probability": 1.0,
      },
    )

  cfg.metrics = {
    "fingertip_contacts": MetricsTermCfg(
      func=hand_cube_mdp.fingertip_contact_count,
      params={"sensor_name": "fingertip_cube_contact"},
    ),
    "yaw_goal_angle": MetricsTermCfg(
      func=hand_cube_mdp.pose_goal_angle_metric,
      params={"command_name": "pose_goal", "object_name": "cube"},
    ),
    "yaw_goal_success": MetricsTermCfg(
      func=hand_cube_mdp.pose_goal_success_metric,
      params={
        "command_name": "pose_goal",
        "object_name": "cube",
        "threshold": math.radians(15.0),
      },
    ),
    "cube_palm_distance": MetricsTermCfg(
      func=hand_cube_mdp.cube_palm_distance_metric,
      params={"object_name": "cube", **_palm_params()},
    ),
  }
  cfg.curriculum = {}
  cfg.episode_length_s = 1e9 if play else 20.0
  cfg.viewer.distance = 0.50
  return cfg
