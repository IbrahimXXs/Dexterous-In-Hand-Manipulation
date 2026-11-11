"""Upstream-reward baseline for the physical right LEAP Hand.

This deliberately calls the shared upstream environment constructor directly
instead of the locally enhanced right-hand wrapper.  It therefore excludes
the four experimental engagement rewards:

* fingertip_participation
* fingertip_proximity
* instant_fingertip_contact
* non_thumb_curl

The right-hand yaw sign correction in the shared constructor is retained.  The
upstream repository omits that correction from its curriculum metric, which
causes the right-hand reward and curriculum to measure opposite directions.
"""

from in_hand_rotation_mjlab.robots import get_leap_right_hand_cfg
from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right.env_cfgs import (
  CUBE_SPAWN_POS,
  GRASP_CACHE_FILE,
  GRASP_INIT_JOINT_POS,
)
from in_hand_rotation_mjlab.tasks.hand_cube.hand_cube_env_cfg import (
  make_hand_cube_embodiment_env_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg


def leap_right_upstream_baseline_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the upstream-reward task with right-hand correctness fixes."""
  return make_hand_cube_embodiment_env_cfg(
    robot_cfg_fn=get_leap_right_hand_cfg,
    grasp_init_joint_pos=GRASP_INIT_JOINT_POS,
    cube_spawn_pos=CUBE_SPAWN_POS,
    grasp_cache_file=GRASP_CACHE_FILE,
    negate_yaw_rate=False,
    play=play,
  )
