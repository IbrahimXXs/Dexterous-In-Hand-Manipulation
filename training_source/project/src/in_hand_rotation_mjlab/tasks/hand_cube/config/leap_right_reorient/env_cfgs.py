"""Arbitrary cube-pose reorientation on the canonical physical right LEAP model."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from in_hand_rotation_mjlab import MYMJLAB_SRC_PATH
from in_hand_rotation_mjlab.tasks.hand_cube import mdp as hand_cube_mdp
from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right.env_cfgs import (
  leap_right_hand_cube_rotate_env_cfg,
)


_CAMERA_POSE_PARAMS = {
  "object_name": "cube",
  "hand_cfg": SceneEntityCfg("robot", body_names=("palm",)),
  "palm_center_geom_expr": "palm_collision_.*",
}


def leap_right_hand_cube_reorient_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build unique SO(3) cube reorientation without changing hardware dynamics.

  The robot XML, identified actuator parameters, action semantics, control and
  physics rates, cube/contact model, domain randomization, and reset states all
  come from the already-validated right-hand rotation task.
  """
  cfg = leap_right_hand_cube_rotate_env_cfg(play=play)

  cfg.commands["pose_goal"] = hand_cube_mdp.CubePoseGoalCommandCfg(
    entity_name="cube",
    resampling_time_range=(1e9, 1e9) if play else (8.0, 12.0),
    success_threshold=math.radians(10.0),
    success_dwell_steps=5,
    resample_on_success=not play,
    target_angle_range=(
      (math.radians(10.0), math.pi)
      if play
      else (math.radians(10.0), math.radians(30.0))
    ),
    debug_vis=play,
  )

  # Use exact, contact-verified policy states and restore hand + cube together.
  # Independent reset jitter would destroy the correspondence between them.
  grasp_reset = cfg.events["reset_from_grasp_cache"].params
  grasp_reset["cache_file"] = str(
    MYMJLAB_SRC_PATH
    / "tasks/hand_cube/cache/leap_right_policy_grasp_cache.npz"
  )
  grasp_reset["scale_list"] = None
  grasp_reset["scale_jitter"] = 0.0
  grasp_reset["pose_range"] = {}
  grasp_reset["restore_robot_joint_state"] = True
  grasp_reset["randomize_labeled_frame"] = True
  cfg.actions["joint_pos"].use_cached_reset_target = True

  cfg.observations["actor"].terms["camera_cube_pose"] = ObservationTermCfg(
    func=hand_cube_mdp.simulated_camera_cube_pose,
    params=_CAMERA_POSE_PARAMS,
    noise=Unoise(n_min=-0.01, n_max=0.01),
    delay_min_lag=0,
    delay_max_lag=2,
    delay_hold_prob=0.9,
    delay_update_period=10,
  )
  pose_goal_params = {
    "command_name": "pose_goal",
    "hand_cfg": _CAMERA_POSE_PARAMS["hand_cfg"],
    "palm_center_geom_expr": _CAMERA_POSE_PARAMS["palm_center_geom_expr"],
  }
  cfg.observations["actor"].terms["pose_goal"] = ObservationTermCfg(
    func=hand_cube_mdp.simulated_camera_pose_goal,
    params=pose_goal_params,
    noise=Unoise(n_min=-0.005, n_max=0.005),
    delay_min_lag=0,
    delay_max_lag=2,
    delay_hold_prob=0.9,
    delay_update_period=10,
  )
  cfg.observations["critic"].terms["camera_cube_pose"] = ObservationTermCfg(
    func=hand_cube_mdp.simulated_camera_cube_pose,
    params=_CAMERA_POSE_PARAMS,
  )
  cfg.observations["critic"].terms["pose_goal"] = ObservationTermCfg(
    func=hand_cube_mdp.simulated_camera_pose_goal,
    params=pose_goal_params,
  )

  engagement_rewards = {
    name: cfg.rewards[name]
    for name in (
      "fingertip_participation",
      "fingertip_proximity",
      "instant_fingertip_contact",
      "non_thumb_curl",
    )
  }
  engagement_rewards["fingertip_participation"].weight = 0.50
  engagement_rewards["fingertip_proximity"].weight = 0.40
  engagement_rewards["instant_fingertip_contact"].weight = 0.10
  engagement_rewards["non_thumb_curl"].weight = 0.10

  cfg.rewards = {
    "pose_alignment": RewardTermCfg(
      func=hand_cube_mdp.pose_alignment_reward,
      weight=8.0,
      params={"command_name": "pose_goal", "object_name": "cube"},
    ),
    "pose_precision": RewardTermCfg(
      func=hand_cube_mdp.pose_precision_reward,
      weight=2.0,
      params={
        "command_name": "pose_goal",
        "object_name": "cube",
        "std": 0.20,
      },
    ),
    "pose_progress": RewardTermCfg(
      func=hand_cube_mdp.pose_alignment_progress,
      weight=4.0,
      params={
        "command_name": "pose_goal",
        "object_name": "cube",
        "clip": 2.0,
      },
    ),
    "pose_success": RewardTermCfg(
      func=hand_cube_mdp.pose_success_reward,
      weight=20.0,
      params={
        "command_name": "pose_goal",
        "object_name": "cube",
        "threshold": math.radians(10.0),
      },
    ),
    "cube_palm_retention": RewardTermCfg(
      func=hand_cube_mdp.cube_palm_retention_reward,
      weight=1.0,
      params={
        **_CAMERA_POSE_PARAMS,
        "safe_distance": 0.13,
        "falloff_distance": 0.03,
      },
    ),
    **engagement_rewards,
    "object_linvel_penalty": RewardTermCfg(
      func=hand_cube_mdp.object_linvel_l1,
      weight=-0.3,
      params={"object_name": "cube"},
    ),
    "pose_diff_penalty": RewardTermCfg(
      func=hand_cube_mdp.pose_diff_l2_from_reset,
      weight=-0.03,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "average_per_joint": False,
        "joint_tolerance": 0.4,
      },
    ),
    "torque_penalty": RewardTermCfg(
      func=hand_cube_mdp.joint_torque_l2,
      weight=-0.05,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "work_penalty": RewardTermCfg(
      func=hand_cube_mdp.actuator_work_l2_penalty,
      weight=-0.03,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "object_fallen": RewardTermCfg(
      func=hand_cube_mdp.object_fallen,
      weight=-100.0,
      params={"object_name": "cube", "minimum_height": 0.2},
    ),
  }

  # Roll/pitch/yaw are all task variables. Terminate only on real grasp loss.
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "cube_fell": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("cube")},
    ),
    "cube_too_fast": TerminationTermCfg(
      func=hand_cube_mdp.object_linear_speed_above,
      params={"max_linear_speed": 1.0, "asset_cfg": SceneEntityCfg("cube")},
    ),
    "cube_palm_distance": TerminationTermCfg(
      func=hand_cube_mdp.object_palm_distance_above,
      params={
        "max_distance": 0.18,
        "object_name": "cube",
        "hand_cfg": _CAMERA_POSE_PARAMS["hand_cfg"],
        "palm_center_geom_expr": _CAMERA_POSE_PARAMS["palm_center_geom_expr"],
      },
    ),
    "nan": TerminationTermCfg(func=envs_mdp.nan_detection),
  }

  cfg.metrics["pose_goal_angle"] = MetricsTermCfg(
    func=hand_cube_mdp.pose_goal_angle_metric,
    params={"command_name": "pose_goal", "object_name": "cube"},
  )
  cfg.metrics["pose_goal_success"] = MetricsTermCfg(
    func=hand_cube_mdp.pose_goal_success_metric,
    params={
      "command_name": "pose_goal",
      "object_name": "cube",
      "threshold": math.radians(10.0),
    },
  )
  cfg.metrics["cube_palm_distance"] = MetricsTermCfg(
    func=hand_cube_mdp.cube_palm_distance_metric,
    params=_CAMERA_POSE_PARAMS,
  )
  cfg.metrics.pop("position_error", None)
  cfg.metrics.pop("rotation_progress", None)
  cfg.metrics.pop("tilt_error", None)

  # Remove rotation-task curricula. Minimum-step gates prevent very early
  # expansion, while strict-success gates stop difficulty from advancing before
  # the current range is genuinely being solved.
  cfg.curriculum = {}
  if not play:
    cfg.curriculum["pose_goal_angle_range"] = CurriculumTermCfg(
      func=hand_cube_mdp.pose_goal_angle_by_success,
      params={
        "command_name": "pose_goal",
        "metric_name": "pose_goal_success",
        "ema_alpha": 0.05,
        "min_steps_per_episode": 40,
        "stages": [
          {
            "minimum_step": 1_600,
            "unlock_success": 0.020,
            "angle_min": math.radians(10.0),
            "angle_max": math.radians(30.0),
          },
          {
            "minimum_step": 4_000,
            "unlock_success": 0.015,
            "angle_min": math.radians(10.0),
            "angle_max": math.radians(45.0),
          },
          {
            "minimum_step": 8_000,
            "unlock_success": 0.0125,
            "angle_min": math.radians(10.0),
            "angle_max": math.radians(60.0),
          },
          {
            "minimum_step": 16_000,
            "unlock_success": 0.010,
            "angle_min": math.radians(10.0),
            "angle_max": math.radians(90.0),
          },
          {
            "minimum_step": 30_000,
            "unlock_success": 0.0075,
            "angle_min": math.radians(10.0),
            "angle_max": math.radians(120.0),
          },
          {
            "minimum_step": 30_000,
            "unlock_success": 1.0,
            "angle_min": math.radians(10.0),
            "angle_max": math.pi,
          },
        ],
      },
    )
  return cfg

