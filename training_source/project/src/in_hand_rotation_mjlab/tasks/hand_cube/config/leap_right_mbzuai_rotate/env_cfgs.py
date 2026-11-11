"""Continuous positive-Z rotation of the 58 mm MBZUAI cube.

This is the original flat, system-identified right-hand rotation task.  It has
no pose target and no visual feedback.  The only task-specific changes are the
measured MBZUAI cube and a coherent grasp cache made for that cube.
"""

from in_hand_rotation_mjlab import MYMJLAB_SRC_PATH
from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_upstream_baseline.env_cfgs import (
  leap_right_upstream_baseline_env_cfg,
)
from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_visual_yaw.env_cfgs import (
  _mbzuai_cube_spec,
)
from mjlab.envs import ManagerBasedRlEnvCfg


_MBZUAI_GRASP_CACHE = "leap_right_mbzuai_58mm_policy_grasp_cache.npz"


def leap_right_mbzuai_rotate_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build continuous Z rotation with the flat right hand and MBZUAI cube."""
  cfg = leap_right_upstream_baseline_env_cfg(play=play)

  # Exact measured collision model: 58 mm edge, 85 g nominal mass.  Training
  # retains the parent task's 0.7--1.4 mass and contact randomization.
  cfg.scene.entities["cube"].spec_fn = _mbzuai_cube_spec

  # Restore paired hand/cube states generated for the larger MBZUAI cube.
  # Independent scale jitter would invalidate those contact-verified grasps.
  grasp_reset = cfg.events["reset_from_grasp_cache"].params
  grasp_reset["cache_file"] = str(
    MYMJLAB_SRC_PATH / "tasks/hand_cube/cache" / _MBZUAI_GRASP_CACHE
  )
  grasp_reset["scale_list"] = None
  grasp_reset["scale_jitter"] = 0.0
  grasp_reset["pose_range"] = {}
  grasp_reset["restore_robot_joint_state"] = True
  grasp_reset["randomize_labeled_frame"] = False
  cfg.actions["joint_pos"].use_cached_reset_target = True

  return cfg
