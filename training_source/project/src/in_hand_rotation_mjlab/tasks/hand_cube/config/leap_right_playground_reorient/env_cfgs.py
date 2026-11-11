"""Playground cube reorientation on the identified physical right LEAP hand."""

import math
from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from in_hand_rotation_mjlab import MYMJLAB_SRC_PATH
from in_hand_rotation_mjlab.tasks.hand_cube import mdp as hand_cube_mdp
from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right.env_cfgs import (
  leap_right_hand_cube_rotate_env_cfg,
)


_MBZUAI_CUBE_ASSET_ROOT = Path(__file__).resolve().parents[6] / "assets/mbzuai_cube"
_PLAYGROUND_CUBE_MESH = _MBZUAI_CUBE_ASSET_ROOT / "dex_cube.obj"
_PLAYGROUND_CUBE_TEXTURE = _MBZUAI_CUBE_ASSET_ROOT / "mbzuai_cube.png"
MBZUAI_CUBE_EDGE_M = 0.058
MBZUAI_CUBE_MASS_KG = 0.085
_CUBE_HALF_SIZE_M = MBZUAI_CUBE_EDGE_M / 2.0
_MBZUAI_GRASP_CACHE = "leap_right_mbzuai_58mm_policy_grasp_cache.npz"

# Playground uses palm quat="0 1 0 -0.175" instead of "0 1 0 0".
# A +Y root rotation by 2*atan(0.175) composes to exactly that palm pose.
PLAYGROUND_NOMINAL_TILT_DEG = math.degrees(2.0 * math.atan(0.175))
PLAYGROUND_TRAIN_TILT_RANGE_DEG = (15.0, 25.0)


def _add_labeled_cube_visual(
  spec: mujoco.MjSpec,
  body: mujoco.MjsBody,
  *,
  name_prefix: str,
  alpha: float = 1.0,
) -> None:
  """Overlay Playground's labeled mesh without adding mass or contacts."""
  missing = [
    path
    for path in (_PLAYGROUND_CUBE_MESH, _PLAYGROUND_CUBE_TEXTURE)
    if not path.is_file()
  ]
  if missing:
    raise FileNotFoundError(
      "MuJoCo Playground labeled-cube assets are missing: "
      + ", ".join(str(path) for path in missing)
    )

  texture_name = f"{name_prefix}_texture"
  material_name = f"{name_prefix}_material"
  mesh_name = f"{name_prefix}_mesh"
  spec.add_texture(
    name=texture_name,
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    file=str(_PLAYGROUND_CUBE_TEXTURE),
  )
  spec.add_material(
    name=material_name,
    textures=["", texture_name],
    rgba=(1.0, 1.0, 1.0, alpha),
  )
  spec.add_mesh(
    name=mesh_name,
    file=str(_PLAYGROUND_CUBE_MESH),
    scale=(_CUBE_HALF_SIZE_M,) * 3,
  )
  body.add_geom(
    name=f"{name_prefix}_visual",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname=mesh_name,
    material=material_name,
    contype=0,
    conaffinity=0,
    density=0.0,
    group=2,
  )


def _goal_cube_spec() -> mujoco.MjSpec:
  """Create the mouse-manipulable, nonphysical target-orientation cube."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="goal_cube")
  _add_labeled_cube_visual(
    spec,
    body,
    name_prefix="goal_cube_label",
    alpha=0.72,
  )
  return spec


def _tracked_cube_spec() -> mujoco.MjSpec:
  """Create a collision-free mocap visual for an external measured pose."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="tracked_cube")
  _add_labeled_cube_visual(
    spec,
    body,
    name_prefix="tracked_cube_label",
    alpha=0.82,
  )
  return spec


def _mbzuai_cube_spec() -> mujoco.MjSpec:
  """Create the measured 58 mm, 85 g MBZUAI cube collision model."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(_CUBE_HALF_SIZE_M,) * 3,
    mass=MBZUAI_CUBE_MASS_KG,
    rgba=(0.85, 0.3, 0.15, 1.0),
  )
  return spec


def _joint_cfg() -> SceneEntityCfg:
  """Return an unresolved selector; regex selectors cannot be safely reused."""
  return SceneEntityCfg("robot", joint_names=(".*",))


def _palm_params() -> dict:
  """Return fresh palm-frame parameters for each manager term."""
  return {
    "object_name": "cube",
    "hand_cfg": SceneEntityCfg("robot", body_names=("palm",)),
    "palm_center_geom_expr": "palm_collision_.*",
  }


def leap_right_playground_reorient_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the public task semantics with this hand's validated dynamics.

  The parent task is deliberately the working right-hand rotation task.  That
  keeps the identified XML/actuators, joint order, contact model, domain
  randomization, 200 Hz physics, and 20 Hz policy loop unchanged. The cube
  nominal and grasp cache are task-specific so they match the physical MBZUAI
  cube instead of altering existing rotation tasks and checkpoints.
  """
  cfg = leap_right_hand_cube_rotate_env_cfg(play=play)
  cfg.scene.entities["cube"].spec_fn = _mbzuai_cube_spec
  if play:
    cube_cfg = cfg.scene.entities["cube"]
    original_cube_spec_fn = cube_cfg.spec_fn

    def _labeled_cube_spec() -> mujoco.MjSpec:
      spec = original_cube_spec_fn()
      cube_body = spec.body("cube")
      _add_labeled_cube_visual(
        spec,
        cube_body,
        name_prefix="cube_label",
      )
      # Hide only the primitive's pixels. Its size, mass, friction, contact
      # type, and name remain the exact collision model used for training.
      spec.geom("cube_geom").rgba[:] = (0.0, 0.0, 0.0, 0.0)
      return spec

    cube_cfg.spec_fn = _labeled_cube_spec
    cfg.scene.entities["goal_cube"] = EntityCfg(
      init_state=EntityCfg.InitialStateCfg(
        pos=(0.16, -0.15, 0.36),
        rot=(1.0, 0.0, 0.0, 0.0),
      ),
      spec_fn=_goal_cube_spec,
    )
    # This visual-only mocap body is parked below the scene until the viewer
    # receives a fresh pose. It cannot change contacts or learned dynamics.
    cfg.scene.entities["tracked_cube"] = EntityCfg(
      init_state=EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, -1.0),
        rot=(1.0, 0.0, 0.0, 0.0),
      ),
      spec_fn=_tracked_cube_spec,
    )
    cfg.viewer.distance = 0.55

  cfg.commands["pose_goal"] = hand_cube_mdp.PlaygroundCubePoseGoalCommandCfg(
    entity_name="cube",
    resampling_time_range=(1e9, 1e9),
    success_threshold=0.1,
    success_dwell_steps=1,
    resample_on_success=False,
    target_angle_range=(0.1, math.pi),
    debug_vis=play,
  )

  # Restore whole, contact-verified hand+cube states rather than destroying
  # their correspondence with independent joint or cube jitter.
  grasp_reset = cfg.events["reset_from_grasp_cache"].params
  grasp_reset["cache_file"] = str(
    MYMJLAB_SRC_PATH
    / "tasks/hand_cube/cache"
    / _MBZUAI_GRASP_CACHE
  )
  grasp_reset["scale_list"] = None
  grasp_reset["scale_jitter"] = 0.0
  grasp_reset["pose_range"] = {}
  grasp_reset["restore_robot_joint_state"] = True
  grasp_reset["randomize_labeled_frame"] = True
  cfg.actions["joint_pos"].use_cached_reset_target = True

  joint_params = {"asset_cfg": _joint_cfg()}
  actor_terms = {
    # The policy server subtracts the identified default pose from raw
    # encoders for this term, matching joint_pos_rel in simulation.
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={**joint_params, "biased": True},
      noise=Unoise(n_min=-0.05, n_max=0.05),
      delay_min_lag=0,
      delay_max_lag=1,
      delay_hold_prob=0.9,
      delay_update_period=10,
    ),
    "joint_pos_error": ObservationTermCfg(
      func=hand_cube_mdp.playground_joint_position_error,
      params={**joint_params, "action_name": "joint_pos", "biased": True},
      noise=Unoise(n_min=-0.05, n_max=0.05),
      delay_min_lag=0,
      delay_max_lag=1,
      delay_hold_prob=0.9,
      delay_update_period=10,
    ),
    "camera_cube_position_error": ObservationTermCfg(
      func=hand_cube_mdp.playground_cube_position_error,
      params=_palm_params(),
      noise=Unoise(n_min=-0.02, n_max=0.02),
      delay_min_lag=0,
      delay_max_lag=2,
      delay_hold_prob=0.9,
      delay_update_period=10,
    ),
    "camera_cube_orientation_error": ObservationTermCfg(
      func=hand_cube_mdp.playground_cube_orientation_error,
      params={**_palm_params(), "command_name": "pose_goal"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
      delay_min_lag=0,
      delay_max_lag=2,
      delay_hold_prob=0.9,
      delay_update_period=10,
    ),
    "last_action": ObservationTermCfg(func=envs_mdp.last_action),
  }
  cfg.observations["actor"] = ObservationGroupCfg(
    actor_terms,
    enable_corruption=not play,
    history_length=None,
    flatten_history_dim=True,
    nan_policy="error",
  )

  # Keep the validated asymmetric critic and add clean task observations.
  critic_terms = dict(cfg.observations["critic"].terms)
  critic_terms["cube_position_error"] = ObservationTermCfg(
    func=hand_cube_mdp.playground_cube_position_error,
    params=_palm_params(),
  )
  critic_terms["cube_orientation_error"] = ObservationTermCfg(
    func=hand_cube_mdp.playground_cube_orientation_error,
    params={**_palm_params(), "command_name": "pose_goal"},
  )
  critic_terms["last_action"] = ObservationTermCfg(func=envs_mdp.last_action)
  cfg.observations["critic"] = ObservationGroupCfg(
    critic_terms,
    enable_corruption=False,
    history_length=1,
    flatten_history_dim=True,
    nan_policy="error",
  )

  cfg.rewards = {
    "orientation": RewardTermCfg(
      func=hand_cube_mdp.playground_orientation_reward,
      weight=5.0,
      params={"command_name": "pose_goal", "object_name": "cube"},
    ),
    "position": RewardTermCfg(
      func=hand_cube_mdp.playground_position_reward,
      weight=0.5,
      params=_palm_params(),
    ),
    "termination": RewardTermCfg(
      func=hand_cube_mdp.object_fallen,
      weight=-100.0,
      params={"object_name": "cube", "minimum_height": 0.2},
    ),
    "hand_pose": RewardTermCfg(
      func=hand_cube_mdp.playground_hand_pose_cost,
      weight=-0.5,
      params={"asset_cfg": _joint_cfg()},
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
    # MJLab scales by dt=0.05: 2000 * 0.05 = Playground's +100 bonus.
    "success": RewardTermCfg(
      func=hand_cube_mdp.pose_success_reward,
      weight=2000.0,
      params={
        "command_name": "pose_goal",
        "object_name": "cube",
        "threshold": 0.1,
      },
    ),
  }

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "cube_fell": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("cube")},
    ),
    "nan": TerminationTermCfg(func=envs_mdp.nan_detection),
  }

  cfg.metrics = {
    name: term
    for name, term in cfg.metrics.items()
    if "contact" in name or "fingertip" in name
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
      "threshold": 0.1,
    },
  )
  cfg.metrics["cube_palm_distance"] = MetricsTermCfg(
    func=hand_cube_mdp.cube_palm_distance_metric,
    params=_palm_params(),
  )

  # Playground trains full SO(3) directly. No curriculum can silently advance
  # before genuine success, which invalidated the previous local experiment.
  cfg.curriculum = {}
  cfg.episode_length_s = 1e9 if play else 50.0
  return cfg


def leap_right_playground_reorient_tilted_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Add the Playground gravity-assisting mount tilt to the SysID task.

  Training randomizes the tilt narrowly around the public Playground value so
  the policy tolerates real mounting error while retaining a consistent gravity
  bias. Play and evaluation use the exact nominal angle. The reset event rotates
  the hand root and cached hand-cube grasp together for every environment.
  """
  cfg = leap_right_playground_reorient_env_cfg(play=play)
  reset = cfg.events["reset_from_grasp_cache"].params
  robot_origin = cfg.scene.entities["robot"].init_state.pos
  if robot_origin is None:
    raise ValueError("The tilted task requires a defined robot root position.")

  nominal_tilt = math.radians(PLAYGROUND_NOMINAL_TILT_DEG)
  if play:
    tilt_range = (nominal_tilt, nominal_tilt)
  else:
    tilt_range = tuple(
      math.radians(angle) for angle in PLAYGROUND_TRAIN_TILT_RANGE_DEG
    )
  reset["world_transform_origin"] = tuple(robot_origin)
  reset["world_transform_pitch_range"] = tilt_range
  return cfg
