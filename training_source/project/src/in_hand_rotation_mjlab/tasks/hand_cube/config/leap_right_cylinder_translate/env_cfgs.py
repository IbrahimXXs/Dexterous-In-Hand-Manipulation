"""Autonomous table pickup and palm-X translation for Task 1."""

from __future__ import annotations

import mujoco

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

from in_hand_rotation_mjlab.tasks.hand_cube import mdp as hand_cube_mdp
from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_grasp.env_cfgs import (
  GRASP_CUBE_SPAWN_XY,
  GRASP_ROBOT_BASE_POS,
  GRASP_ROBOT_BASE_ROT,
  TABLE_HEIGHT,
  leap_right_hand_cube_grasp_env_cfg,
)


CYLINDER_RADIUS_M = 0.0325
CYLINDER_HALF_LENGTH_M = 0.0375
CYLINDER_MASS_KG = 0.10
CYLINDER_SPAWN_XY = GRASP_CUBE_SPAWN_XY
PALM_DOWN_ROBOT_BASE_POS = GRASP_ROBOT_BASE_POS
PALM_DOWN_ROBOT_BASE_ROT_WXYZ = GRASP_ROBOT_BASE_ROT

TARGET_CLEARANCE = 0.008
PICKUP_APPROACH_STEPS = 48
PICKUP_LIFT_STEPS = 40
PICKUP_SECURE_STEPS = 30
CYLINDER_APPROACH_JOINT_POS = (
  0.12675853, -0.48722458, 0.82711178, 1.66576707,
  -0.02074096, -0.07140364, 0.88666087, 1.92238200,
  0.22822130, 0.08002567, 0.28994086, 0.14048380,
  1.67592859, -0.46332616, 0.87497735, 1.67515814,
)
CYLINDER_LIFT_JOINT_POS = (
  0.13696332, -0.37278649, 1.06866598, 1.91408551,
  -0.11429206, -0.38869587, 1.37741387, 1.92535722,
  0.22800924, -0.04182740, 0.29680836, 0.15549476,
  1.33377051, -0.07399283, 0.59060264, 1.58691752,
)
CYLINDER_SECURE_JOINT_POS = (
  -0.03423974, -0.40250498, 1.06342125, 1.90610254,
  -0.12686770, -0.38259897, 1.15717840, 1.91414213,
  0.25568551, 0.04460016, 0.37459743, 0.19611788,
  1.38331962, -0.13754722, 0.91532135, 1.57553756,
)

# Canonical physical-right LEAP home pose, in actuator/joint order.
CYLINDER_OPEN_JOINT_POS = (
  0.3, 0.0, 0.3, 0.2,
  0.3, 0.0, 0.3, 0.2,
  0.3, 0.0, 0.3, 0.2,
  0.4, 0.2, 0.3, 0.2,
)


def _cylinder_spec() -> mujoco.MjSpec:
  """Measured upright 65 mm diameter by 75 mm cylinder."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=(CYLINDER_RADIUS_M, CYLINDER_HALF_LENGTH_M, 0.0),
    mass=CYLINDER_MASS_KG,
    friction=(1.0, 0.01, 0.001),
    rgba=(0.20, 0.75, 0.30, 1.0),
  )
  return spec


def _cylinder_cfg() -> SceneEntityCfg:
  return SceneEntityCfg(
    "cube", geom_names=["cube_geom"], preserve_order=True
  )


def _fingertip_cfg() -> SceneEntityCfg:
  return SceneEntityCfg(
    "robot",
    geom_names=["if_tip", "mf_tip", "rf_tip", "th_tip"],
    preserve_order=True,
  )


def _palm_params() -> dict:
  return {
    "object_name": "cube",
    "hand_cfg": SceneEntityCfg("robot", body_names=("palm",)),
    "palm_center_geom_expr": "palm_collision_.*",
  }


def _cylinder_table_reset(play: bool) -> EventTermCfg:
  return EventTermCfg(
    func=hand_cube_mdp.reset_cylinder_on_table,
    mode="reset",
    params={
      "nominal_spawn_xy": CYLINDER_SPAWN_XY,
      "xy_jitter": (0.0, 0.0) if play else (-0.003, 0.003),
      "yaw_range": (-3.14159, 3.14159),
      "table_height": TABLE_HEIGHT,
      "spawn_clearance": 0.0,
      "scale_list": (1.0,),
      "scale_jitter": 0.0,
      "cylinder_cfg": _cylinder_cfg(),
    },
  )


def leap_right_hand_cylinder_pickup_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Stage A: open, grasp, and lift a real cylinder from the table."""
  cfg = leap_right_hand_cube_grasp_env_cfg(play=play)
  cfg.scene.entities["cube"].spec_fn = _cylinder_spec
  cfg.scene.entities["cube"].init_state.pos = (
    CYLINDER_SPAWN_XY[0],
    CYLINDER_SPAWN_XY[1],
    TABLE_HEIGHT + CYLINDER_HALF_LENGTH_M,
  )
  cfg.scene.entities["cube"].init_state.rot = (1.0, 0.0, 0.0, 0.0)

  events = {}
  for name, term in cfg.events.items():
    if name == "reset_cube_on_table":
      events["reset_cylinder_on_table"] = _cylinder_table_reset(play)
    else:
      events[name] = term
  cfg.events = events

  cfg.curriculum.pop("lower_cube_to_table", None)

  cylinder_cfg = _cylinder_cfg()
  fingertip_cfg = _fingertip_cfg()
  cfg.rewards["instant_fingertip_contact"].weight = 0.5
  cfg.rewards["fingertip_participation"].weight = 0.3
  cfg.rewards["fingertip_proximity"] = RewardTermCfg(
    func=hand_cube_mdp.cylinder_fingertip_proximity,
    weight=0.5,
    params={
      "hand_cfg": fingertip_cfg,
      "cylinder_cfg": cylinder_cfg,
      "surface_margin_m": 0.006,
      "distance_scale_m": 0.030,
      "worst_tip_weight": 0.45,
    },
  )
  cfg.rewards["lift_progress"] = RewardTermCfg(
    func=hand_cube_mdp.contact_gated_cylinder_lift_progress,
    weight=20.0,
    params={
      "sensor_name": "fingertip_cube_contact",
      "target_clearance": TARGET_CLEARANCE,
      "minimum_contacts": 2,
      "table_height": TABLE_HEIGHT,
      "cylinder_cfg": cylinder_cfg,
    },
  )
  cfg.rewards["sustained_lift"] = RewardTermCfg(
    func=hand_cube_mdp.sustained_cylinder_grasp_lift,
    weight=30.0,
    params={
      "sensor_name": "fingertip_cube_contact",
      "target_clearance": TARGET_CLEARANCE,
      "minimum_contacts": 2,
      "table_height": TABLE_HEIGHT,
      "cylinder_cfg": cylinder_cfg,
    },
  )
  cfg.rewards["cylinder_upright"] = RewardTermCfg(
    func=hand_cube_mdp.cylinder_upright_reward,
    weight=0.8,
    params={"cylinder_cfg": cylinder_cfg, "std_rad": 0.25},
  )

  cfg.commands["pickup_phase"] = (
    hand_cube_mdp.CylinderTablePickupCommandCfg(
      entity_name="cube",
      hand_name="robot",
      sensor_name="fingertip_cube_contact",
      geom_name="cube_geom",
      translation_command_name=None,
      open_joint_pos=CYLINDER_OPEN_JOINT_POS,
      table_height_m=TABLE_HEIGHT,
      target_clearance_m=TARGET_CLEARANCE,
      table_recovery_clearance_m=0.004,
      maximum_table_speed_mps=0.10,
      minimum_contacts=2,
      recovery_spawn_xy=CYLINDER_SPAWN_XY,
      recovery_xy_jitter=(0.0, 0.0) if play else (-0.003, 0.003),
      recovery_yaw_range=(-3.14159, 3.14159),
      open_joint_tolerance_rad=0.12,
      open_dwell_steps=8,
      acquire_dwell_steps=40,
      minimum_pickup_steps=(
        PICKUP_APPROACH_STEPS + PICKUP_LIFT_STEPS + PICKUP_SECURE_STEPS
      ),
      table_dwell_steps=5,
      resampling_time_range=(1e9, 1e9),
      debug_vis=False,
    )
  )
  cfg.observations["actor"].terms["pickup_phase"] = ObservationTermCfg(
    func=envs_mdp.generated_commands,
    params={"command_name": "pickup_phase"},
  )
  cfg.observations["critic"].terms["pickup_phase"] = ObservationTermCfg(
    func=envs_mdp.generated_commands,
    params={"command_name": "pickup_phase"},
  )
  total_pickup_steps = (
    PICKUP_APPROACH_STEPS + PICKUP_LIFT_STEPS + PICKUP_SECURE_STEPS
  )
  cfg.observations["actor"].terms["pickup_progress"] = ObservationTermCfg(
    func=hand_cube_mdp.pickup_progress_observation,
    params={
      "command_name": "pickup_phase",
      "total_pickup_steps": total_pickup_steps,
    },
  )
  cfg.observations["critic"].terms["pickup_progress"] = ObservationTermCfg(
    func=hand_cube_mdp.pickup_progress_observation,
    params={
      "command_name": "pickup_phase",
      "total_pickup_steps": total_pickup_steps,
    },
  )
  cfg.rewards["open_pose"] = RewardTermCfg(
    func=hand_cube_mdp.pickup_open_pose_reward,
    weight=4.0,
    params={"command_name": "pickup_phase", "std_rad": 0.20},
  )
  cfg.rewards["pickup_joint_pose"] = RewardTermCfg(
    func=hand_cube_mdp.pickup_joint_pose_reward,
    weight=10.0,
    params={
      "command_name": "pickup_phase",
      "open_joint_pos": CYLINDER_OPEN_JOINT_POS,
      "approach_joint_pos": CYLINDER_APPROACH_JOINT_POS,
      "lift_joint_pos": CYLINDER_LIFT_JOINT_POS,
      "secure_joint_pos": CYLINDER_SECURE_JOINT_POS,
      "approach_steps": PICKUP_APPROACH_STEPS,
      "lift_steps": PICKUP_LIFT_STEPS,
      "secure_steps": PICKUP_SECURE_STEPS,
      "std_rad": 0.25,
    },
  )
  cfg.rewards["pickup_reference_action"] = RewardTermCfg(
    func=hand_cube_mdp.pickup_reference_action_reward,
    weight=40.0,
    params={
      "command_name": "pickup_phase",
      "action_name": "joint_pos",
      "open_joint_pos": CYLINDER_OPEN_JOINT_POS,
      "approach_joint_pos": CYLINDER_APPROACH_JOINT_POS,
      "lift_joint_pos": CYLINDER_LIFT_JOINT_POS,
      "secure_joint_pos": CYLINDER_SECURE_JOINT_POS,
      "approach_steps": PICKUP_APPROACH_STEPS,
      "lift_steps": PICKUP_LIFT_STEPS,
      "secure_steps": PICKUP_SECURE_STEPS,
      "maximum_command_delta_rad": 1.0 / 24.0,
      "std": 0.30,
    },
  )
  cfg.rewards["pickup_completion"] = RewardTermCfg(
    func=hand_cube_mdp.pickup_completion_reward,
    weight=200.0,
    params={"command_name": "pickup_phase"},
  )
  cfg.events["return_cylinder_to_table"] = EventTermCfg(
    func=hand_cube_mdp.place_cylinder_on_table_for_recovery,
    mode="interval",
    interval_range_s=((12.0, 12.0) if play else (8.0, 12.0)),
    is_global_time=False,
    params={
      "command_name": "pickup_phase",
      "nominal_spawn_xy": CYLINDER_SPAWN_XY,
      "xy_jitter": (0.0, 0.0) if play else (-0.003, 0.003),
      "yaw_range": (-3.14159, 3.14159),
      "table_height": TABLE_HEIGHT,
      "cylinder_cfg": cylinder_cfg,
      "probability": 1.0,
    },
  )

  cfg.metrics["table_clearance"] = MetricsTermCfg(
    func=hand_cube_mdp.cylinder_bottom_clearance,
    params={"table_height": TABLE_HEIGHT, "cylinder_cfg": cylinder_cfg},
  )
  cfg.metrics["grasp_success"] = MetricsTermCfg(
    func=hand_cube_mdp.cylinder_grasp_success_metric,
    params={
      "sensor_name": "fingertip_cube_contact",
      "target_clearance": TARGET_CLEARANCE,
      "minimum_contacts": 2,
      "table_height": TABLE_HEIGHT,
      "cylinder_cfg": cylinder_cfg,
    },
  )
  cfg.metrics["pickup_phase"] = MetricsTermCfg(
    func=hand_cube_mdp.pickup_phase_metric,
    params={"command_name": "pickup_phase"},
  )
  cfg.metrics["completed_pickups"] = MetricsTermCfg(
    func=hand_cube_mdp.completed_pickups_metric,
    params={"command_name": "pickup_phase"},
  )

  cfg.episode_length_s = 1e9 if play else 30.0
  cfg.viewer.distance = 0.42
  cfg.viewer.elevation = -30
  cfg.viewer.azimuth = 135
  return cfg


def leap_right_hand_cylinder_pickup_translate_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Final Task 1: table pickup, verified lift, then palm-X translation."""
  cfg = leap_right_hand_cylinder_pickup_env_cfg(play=play)
  positive_range = (0.050, 0.050) if play else (0.006, 0.008)
  negative_range = (0.020, 0.020) if play else (0.006, 0.008)
  cfg.commands["translation_goal"] = (
    hand_cube_mdp.ObjectTranslationGoalCommandCfg(
      entity_name="cube",
      hand_name="robot",
      palm_body_name="palm",
      resampling_time_range=(1e9, 1e9),
      positive_distance_range=positive_range,
      negative_distance_range=negative_range,
      success_threshold=0.003,
      success_dwell_steps=5,
      resample_on_success=True,
      debug_vis=False,
    )
  )
  cfg.commands["pickup_phase"].translation_command_name = "translation_goal"

  translation_obs = ObservationTermCfg(
    func=hand_cube_mdp.translation_goal_error_palm,
    params={
      **_palm_params(),
      "command_name": "translation_goal",
    },
    noise=None if play else Unoise(n_min=-0.0015, n_max=0.0015),
    scale=100.0,
    delay_min_lag=0,
    delay_max_lag=0 if play else 2,
    delay_hold_prob=0.9,
    delay_update_period=10,
  )
  # Term-major history means this 30-value block is appended after the complete
  # 450-input pickup actor, so checkpoint widening preserves all source columns.
  cfg.observations["actor"].terms["translation_goal_error"] = translation_obs
  cfg.observations["critic"].terms["translation_goal_error"] = ObservationTermCfg(
    func=hand_cube_mdp.translation_goal_error_palm,
    params={
      **_palm_params(),
      "command_name": "translation_goal",
    },
    scale=100.0,
  )

  cfg.rewards.update(
    {
      "translation_axis_alignment": RewardTermCfg(
        func=hand_cube_mdp.translation_axis_alignment_reward,
        weight=30.0,
        params={
          "command_name": "translation_goal",
          "object_name": "cube",
          "fraction_std": 0.35,
        },
      ),
      "translation_alignment": RewardTermCfg(
        func=hand_cube_mdp.translation_alignment_reward,
        weight=1.0,
        params={
          "command_name": "translation_goal",
          "object_name": "cube",
          "std": 0.007,
        },
      ),
      "translation_progress": RewardTermCfg(
        func=hand_cube_mdp.translation_axis_progress_reward,
        weight=3.0,
        params={
          "command_name": "translation_goal",
          "object_name": "cube",
          "max_rate": 5.0,
        },
      ),
      "translation_success": RewardTermCfg(
        func=hand_cube_mdp.translation_success_reward,
        weight=50.0,
        params={
          "command_name": "translation_goal",
          "object_name": "cube",
          "threshold": 0.003,
        },
      ),
      "translation_completion": RewardTermCfg(
        func=hand_cube_mdp.translation_goal_completion_reward,
        weight=200.0,
        params={"command_name": "translation_goal"},
      ),
      "cross_axis_drift": RewardTermCfg(
        func=hand_cube_mdp.translation_excess_cross_axis_fraction,
        weight=-5.0,
        params={
          "command_name": "translation_goal",
          "object_name": "cube",
          "allowed_fraction": 0.5,
          "max_excess": 2.0,
        },
      ),
    }
  )
  cfg.terminations["cylinder_tilted_during_translation"] = TerminationTermCfg(
    func=hand_cube_mdp.manipulating_cylinder_tilt_above,
    params={
      "command_name": "pickup_phase",
      "cylinder_cfg": _cylinder_cfg(),
      "max_angle_rad": 0.75,
    },
  )
  cfg.metrics["translation_goal_distance"] = MetricsTermCfg(
    func=hand_cube_mdp.translation_goal_distance,
    params={"command_name": "translation_goal", "object_name": "cube"},
  )
  cfg.metrics["translation_goal_success"] = MetricsTermCfg(
    func=hand_cube_mdp.translation_goal_success_metric,
    params={
      "command_name": "translation_goal",
      "object_name": "cube",
      "threshold": 0.003,
    },
  )

  if not play:
    cfg.curriculum["translation_distance"] = CurriculumTermCfg(
      func=hand_cube_mdp.translation_distance_by_success,
      params={
        "command_name": "translation_goal",
        "metric_name": "translation_goal_success",
        "ema_alpha": 0.05,
        "min_steps_per_episode": 40,
        "stages": [
          {
            "minimum_step": 4_000,
            "unlock_success": 0.05,
            "positive_min": 0.006,
            "positive_max": 0.008,
            "negative_min": 0.006,
            "negative_max": 0.008,
          },
          {
            "minimum_step": 8_000,
            "unlock_success": 0.10,
            "positive_min": 0.008,
            "positive_max": 0.020,
            "negative_min": 0.006,
            "negative_max": 0.012,
          },
          {
            "minimum_step": 12_000,
            "unlock_success": 0.20,
            "positive_min": 0.015,
            "positive_max": 0.035,
            "negative_min": 0.010,
            "negative_max": 0.017,
          },
          {
            "minimum_step": 16_000,
            "unlock_success": 0.40,
            "positive_min": 0.030,
            "positive_max": 0.050,
            "negative_min": 0.015,
            "negative_max": 0.020,
          },
        ],
      },
    )

  cfg.episode_length_s = 1e9 if play else 40.0
  return cfg
