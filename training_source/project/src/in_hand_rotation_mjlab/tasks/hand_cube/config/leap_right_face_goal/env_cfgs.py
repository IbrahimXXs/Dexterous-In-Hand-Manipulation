"""Camera-ready numbered-face reorientation using the canonical right LEAP model."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
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


def leap_right_hand_cube_face_up_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build the face-up task without changing the validated hardware dynamics.

  Robot, actuator/SysID parameters, cube dimensions/mass, contact settings,
  grasp-cache reset, action representation, timing, and domain randomization
  are inherited directly from ``leap_right_hand_cube_rotate_env_cfg``.
  """
  cfg = leap_right_hand_cube_rotate_env_cfg(play=play)

  cfg.commands["face_goal"] = hand_cube_mdp.FaceUpCommandCfg(
    entity_name="cube",
    resampling_time_range=(1e9, 1e9) if play else (8.0, 12.0),
    success_threshold=0.20944,
    success_dwell_steps=5,
    # Stage one teaches observable 90-degree face transitions. The command
    # reads raw free-joint qpos, so current-face exclusion is reset-safe.
    exclude_current_face=True,
    target_angle_range=(1.0472, 2.0944),
    resample_on_success=not play,
    debug_vis=False,
  )

  # Use contact-verified rollout states, not the passive-settling cache (which
  # accepts palm support without fingertip contact). Restore the cached hand and
  # cube atomically. Independent position/yaw/size jitter would invalidate that
  # match; label diversity instead comes from the 24 exact cube symmetries.
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

  camera_pose_params = {
    "object_name": "cube",
    "hand_cfg": SceneEntityCfg("robot", body_names=("palm",)),
    "palm_center_geom_expr": "palm_collision_.*",
  }
  cfg.observations["actor"].terms["camera_cube_pose"] = ObservationTermCfg(
    func=hand_cube_mdp.simulated_camera_cube_pose,
    params=camera_pose_params,
    noise=Unoise(n_min=-0.01, n_max=0.01),
    # Model a 0--100 ms pose-estimator delay at the 20 Hz policy rate.
    delay_min_lag=0,
    delay_max_lag=2,
    delay_hold_prob=0.9,
    delay_update_period=10,
  )
  face_goal_params = {
    **camera_pose_params,
    "command_name": "face_goal",
  }
  cfg.observations["actor"].terms["face_goal"] = ObservationTermCfg(
    func=hand_cube_mdp.simulated_camera_face_goal,
    params=face_goal_params,
    noise=Unoise(n_min=-0.01, n_max=0.01),
    delay_min_lag=0,
    delay_max_lag=2,
    delay_hold_prob=0.9,
    delay_update_period=10,
  )
  cfg.observations["critic"].terms["camera_cube_pose"] = ObservationTermCfg(
    func=hand_cube_mdp.simulated_camera_cube_pose,
    params=camera_pose_params,
  )
  cfg.observations["critic"].terms["face_goal"] = ObservationTermCfg(
    func=hand_cube_mdp.simulated_camera_face_goal,
    params=face_goal_params,
  )

  # Preserve the right-hand engagement terms verbatim from the canonical task.
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
    "face_alignment": RewardTermCfg(
      func=hand_cube_mdp.face_alignment_reward,
      weight=8.0,
      params={"command_name": "face_goal", "object_name": "cube"},
    ),
    "face_precision": RewardTermCfg(
      func=hand_cube_mdp.face_precision_reward,
      weight=1.0,
      params={
        "command_name": "face_goal",
        "object_name": "cube",
        "std": 0.20,
      },
    ),
    "face_progress": RewardTermCfg(
      func=hand_cube_mdp.face_alignment_progress,
      weight=4.0,
      params={
        "command_name": "face_goal",
        "object_name": "cube",
        "clip": 2.0,
      },
    ),
    "face_success": RewardTermCfg(
      func=hand_cube_mdp.face_success_reward,
      weight=20.0,
      params={
        "command_name": "face_goal",
        "object_name": "cube",
        "threshold": 0.20944,
      },
    ),
    "cube_palm_retention": RewardTermCfg(
      func=hand_cube_mdp.cube_palm_retention_reward,
      weight=1.0,
      params={
        **camera_pose_params,
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

  # Full roll/pitch motion is the task, so only translation is bounded here.
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
        # Canonical valid grasps are around 0.06--0.10 m after the shared-pose
        # mutation fix; leave margin for regrasp without masking a real drop.
        "max_distance": 0.18,
        "object_name": "cube",
        "hand_cfg": SceneEntityCfg("robot", body_names=("palm",)),
        "palm_center_geom_expr": "palm_collision_.*",
      },
    ),
    "nan": TerminationTermCfg(func=envs_mdp.nan_detection),
  }

  cfg.metrics["face_goal_angle"] = MetricsTermCfg(
    func=hand_cube_mdp.face_goal_angle_metric,
    params={"command_name": "face_goal", "object_name": "cube"},
  )
  cfg.metrics["face_goal_success"] = MetricsTermCfg(
    func=hand_cube_mdp.face_goal_success_metric,
    params={
      "command_name": "face_goal",
      "object_name": "cube",
      "threshold": 0.20944,
    },
  )
  cfg.metrics["cube_palm_distance"] = MetricsTermCfg(
    func=hand_cube_mdp.cube_palm_distance_metric,
    params={
      "object_name": "cube",
      "hand_cfg": SceneEntityCfg("robot", body_names=("palm",)),
      "palm_center_geom_expr": "palm_collision_.*",
    },
  )
  cfg.metrics.pop("position_error", None)
  cfg.metrics.pop("rotation_progress", None)
  cfg.metrics.pop("tilt_error", None)

  # Rotation-task curricula refer to rewards/metrics replaced above.
  cfg.curriculum = {}
  return cfg
