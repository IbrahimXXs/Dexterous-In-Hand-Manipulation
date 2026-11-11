"""Camera-ready table pickup using the canonical right LEAP SysID model."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from in_hand_rotation_mjlab.robots import get_leap_right_hand_cfg
from in_hand_rotation_mjlab.tasks.hand_cube import mdp as hand_cube_mdp
from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right.env_cfgs import (
  leap_right_hand_cube_rotate_env_cfg,
)


# External task setup. The mocap wrapper rotates the canonical palm-up model by
# 180 degrees about X, producing the same top-down palm frame as the legacy
# grasp demo. The resulting palm-body origin is (0, 0.04, 0.1225) m.
GRASP_ROBOT_BASE_POS = (0.0, 0.04, 0.2225)
GRASP_ROBOT_BASE_ROT = (0.0, 1.0, 0.0, 0.0)
GRASP_CUBE_SPAWN_XY = (0.0, 0.04)
NOMINAL_CUBE_HALF_SIZE = 0.0375
TABLE_HEIGHT = 0.0
TARGET_CLEARANCE = 0.020
TRAINING_INITIAL_SPAWN_CLEARANCE = 0.012


def leap_right_hand_cube_grasp_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build a true table-pickup task without changing hand dynamics.

  Actuators, SysID gains, limits, delays, contacts, cube mass/geometry, action
  representation, 20 Hz policy timing, and domain randomization are inherited
  from the canonical right-hand rotation configuration. Only external mounting,
  reset state, sensing, reward, and termination logic are task-specific.
  """
  cfg = leap_right_hand_cube_rotate_env_cfg(play=play)

  robot_cfg = cfg.scene.entities["robot"]
  open_robot_cfg = get_leap_right_hand_cfg()
  robot_cfg.init_state.pos = GRASP_ROBOT_BASE_POS
  robot_cfg.init_state.rot = GRASP_ROBOT_BASE_ROT
  robot_cfg.init_state.joint_pos = dict(open_robot_cfg.init_state.joint_pos or {})

  cube_cfg = cfg.scene.entities["cube"]
  cube_cfg.init_state.pos = (
    GRASP_CUBE_SPAWN_XY[0],
    GRASP_CUBE_SPAWN_XY[1],
    TABLE_HEIGHT + NOMINAL_CUBE_HALF_SIZE,
  )
  cube_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)

  # Start from the physical home pose with small encoder/setup robustness in
  # training. Hardware play/deployment remains deterministic.
  cfg.events["reset_robot_joints"].params["position_range"] = (
    (0.0, 0.0) if play else (-0.03, 0.03)
  )
  cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)

  # Replace the in-hand grasp cache with an atomic table reset. The cube really
  # rests on the plane in play; training begins 12 mm high and performance-based
  # curriculum lowers that clearance smoothly to zero.
  table_reset = EventTermCfg(
    func=hand_cube_mdp.reset_cube_on_table,
    mode="reset",
    params={
      "nominal_spawn_xy": GRASP_CUBE_SPAWN_XY,
      "xy_jitter": (0.0, 0.0) if play else (-0.01, 0.01),
      "yaw_range": (-3.14159, 3.14159),
      "table_height": TABLE_HEIGHT,
      "spawn_clearance": (
        0.0 if play else TRAINING_INITIAL_SPAWN_CLEARANCE
      ),
      "scale_list": (1.0,) if play else (0.95, 0.9, 1.0, 1.05, 1.1),
      "scale_jitter": 0.0 if play else 0.025,
      "cube_cfg": SceneEntityCfg("cube", geom_names=("cube_geom",)),
    },
  )
  grasp_events = {}
  for name, term in cfg.events.items():
    if name == "reset_cube_pose":
      continue
    if name == "reset_from_grasp_cache":
      grasp_events["reset_cube_on_table"] = table_reset
    else:
      grasp_events[name] = term
  cfg.events = grasp_events

  # Camera-ready actor: in hardware this 9D pose is supplied by the calibrated
  # external camera pose estimator. Simulation injects realistic noise/delay.
  camera_pose_params = {
    "object_name": "cube",
    "hand_cfg": SceneEntityCfg("robot", body_names=("palm",)),
    "palm_center_geom_expr": "palm_collision_.*",
  }
  cfg.observations["actor"].terms["camera_cube_pose"] = ObservationTermCfg(
    func=hand_cube_mdp.simulated_camera_cube_pose,
    params=camera_pose_params,
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

  fingertip_cfg = SceneEntityCfg(
    "robot",
    geom_names=("if_tip", "mf_tip", "rf_tip", "th_tip"),
    preserve_order=True,
  )
  cube_geom_cfg = SceneEntityCfg(
    "cube", geom_names=("cube_geom",), preserve_order=True
  )

  # Unlike the legacy policy, lift credit is gated by >=2 fingertip contacts
  # and measures the lowest oriented cube corner, not merely cube-center Z.
  cfg.rewards = {
    "fingertip_proximity": RewardTermCfg(
      func=hand_cube_mdp.fingertip_cube_proximity,
      weight=0.6,
      params={
        "hand_cfg": fingertip_cfg,
        "cube_cfg": cube_geom_cfg,
        "surface_margin": 0.015,
        "distance_scale": 0.030,
        "worst_tip_weight": 0.45,
      },
    ),
    "instant_fingertip_contact": RewardTermCfg(
      func=hand_cube_mdp.fingertip_cube_contact_fraction,
      weight=0.8,
      params={
        "sensor_name": "fingertip_cube_contact",
        "max_contacts": 4,
      },
    ),
    "fingertip_participation": RewardTermCfg(
      func=hand_cube_mdp.fingertip_contact_participation,
      weight=0.4,
      params={
        "sensor_name": "fingertip_cube_contact",
        "window_steps": 40,
        "expected_fingertips": 4,
        "participation_power": 2.0,
      },
    ),
    "lift_progress": RewardTermCfg(
      func=hand_cube_mdp.contact_gated_lift_progress,
      weight=4.0,
      params={
        "sensor_name": "fingertip_cube_contact",
        "target_clearance": TARGET_CLEARANCE,
        "minimum_contacts": 2,
        "table_height": TABLE_HEIGHT,
        "cube_cfg": cube_geom_cfg,
      },
    ),
    "sustained_lift": RewardTermCfg(
      func=hand_cube_mdp.sustained_grasp_lift,
      weight=6.0,
      params={
        "sensor_name": "fingertip_cube_contact",
        "target_clearance": TARGET_CLEARANCE,
        "minimum_contacts": 2,
        "table_height": TABLE_HEIGHT,
        "cube_cfg": cube_geom_cfg,
      },
    ),
    "non_thumb_curl": RewardTermCfg(
      func=hand_cube_mdp.non_thumb_finger_curl,
      weight=0.10,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=("if_pip", "mf_pip", "rf_pip"),
          preserve_order=True,
        ),
        "open_angle": 0.35,
        "engaged_angle": 0.90,
        "worst_finger_weight": 0.5,
      },
    ),
    "xy_drift_penalty": RewardTermCfg(
      func=hand_cube_mdp.cube_xy_drift_penalty,
      weight=-1.5,
      params={
        "nominal_spawn_xy": GRASP_CUBE_SPAWN_XY,
        "deadzone": 0.02,
        "full_penalty_distance": 0.10,
        "cube_cfg": cube_geom_cfg,
      },
    ),
    "object_speed_penalty": RewardTermCfg(
      func=hand_cube_mdp.object_linvel_l1,
      weight=-0.05,
      params={
        "object_name": "cube",
        "clip_max": 1.0,
        "normalize": True,
      },
    ),
    "torque_penalty": RewardTermCfg(
      func=hand_cube_mdp.joint_torque_l2,
      weight=-0.02,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "work_penalty": RewardTermCfg(
      func=hand_cube_mdp.actuator_work_l2_penalty,
      weight=-0.01,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "max_abs_power": 1.0,
        "normalize": True,
      },
    ),
  }

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "cube_below_table": TerminationTermCfg(
      func=hand_cube_mdp.cube_below_table,
      params={
        "minimum_center_height": -0.01,
        "cube_cfg": cube_geom_cfg,
      },
    ),
    "cube_out_of_workspace": TerminationTermCfg(
      func=hand_cube_mdp.cube_xy_drift_above,
      params={
        "nominal_spawn_xy": GRASP_CUBE_SPAWN_XY,
        "maximum_distance": 0.15,
        "cube_cfg": cube_geom_cfg,
      },
    ),
    "cube_too_fast": TerminationTermCfg(
      func=hand_cube_mdp.object_linear_speed_above,
      params={"max_linear_speed": 2.0, "asset_cfg": SceneEntityCfg("cube")},
    ),
    "nan": TerminationTermCfg(func=envs_mdp.nan_detection),
  }

  cfg.metrics = {
    "table_clearance": MetricsTermCfg(
      func=hand_cube_mdp.cube_bottom_clearance,
      params={"table_height": TABLE_HEIGHT, "cube_cfg": cube_geom_cfg},
    ),
    "grasp_success": MetricsTermCfg(
      func=hand_cube_mdp.grasp_success_metric,
      params={
        "sensor_name": "fingertip_cube_contact",
        "target_clearance": TARGET_CLEARANCE,
        "minimum_contacts": 2,
        "table_height": TABLE_HEIGHT,
        "cube_cfg": cube_geom_cfg,
      },
    ),
    "xy_drift": MetricsTermCfg(
      func=hand_cube_mdp.cube_xy_distance_from_spawn,
      params={
        "nominal_spawn_xy": GRASP_CUBE_SPAWN_XY,
        "cube_cfg": cube_geom_cfg,
      },
    ),
    "linear_speed": MetricsTermCfg(
      func=hand_cube_mdp.object_linear_speed,
      params={"asset_cfg": SceneEntityCfg("cube")},
    ),
  }
  for fingertip_index, fingertip_name in enumerate(("if", "mf", "rf", "th")):
    cfg.metrics[f"{fingertip_name}_contact"] = MetricsTermCfg(
      func=hand_cube_mdp.fingertip_contact,
      params={
        "sensor_name": "fingertip_cube_contact",
        "fingertip_index": fingertip_index,
      },
    )

  cfg.curriculum = {}
  if not play:
    cfg.curriculum["lower_cube_to_table"] = CurriculumTermCfg(
      func=hand_cube_mdp.event_param_by_metric_progress,
      params={
        "event_name": "reset_cube_on_table",
        "param_name": "spawn_clearance",
        "metric_name": "grasp_success",
        "progress_min": 0.05,
        "progress_max": 0.40,
        "value_min": 0.0,
        "value_max": TRAINING_INITIAL_SPAWN_CLEARANCE,
        "invert_metric": True,
        "ema_alpha": 0.08,
        "value_lerp": 0.20,
        "min_steps_per_episode": 20,
      },
    )

  cfg.viewer.distance = 0.42
  cfg.viewer.elevation = -35
  cfg.viewer.azimuth = 135
  return cfg
