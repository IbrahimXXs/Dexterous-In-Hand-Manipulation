"""Hardware-aligned 60-second right-hand cube-rotation task.

The reward set and domain randomization come directly from the upstream
baseline constructor. This variant changes only the robot joint ranges and
training episode duration; it retains the existing right-hand SysID model.
"""

from in_hand_rotation_mjlab.robots import get_leap_right_hardware_hand_cfg
from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right.env_cfgs import (
    CUBE_SPAWN_POS,
    GRASP_CACHE_FILE,
    GRASP_INIT_JOINT_POS,
)
from in_hand_rotation_mjlab.tasks.hand_cube.hand_cube_env_cfg import (
    make_hand_cube_embodiment_env_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg


def leap_right_hardware_60s_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the physical-limit task with a 60-second training horizon."""
    cfg = make_hand_cube_embodiment_env_cfg(
        robot_cfg_fn=get_leap_right_hardware_hand_cfg,
        grasp_init_joint_pos=GRASP_INIT_JOINT_POS,
        cube_spawn_pos=CUBE_SPAWN_POS,
        grasp_cache_file=GRASP_CACHE_FILE,
        negate_yaw_rate=False,
        play=play,
    )
    if not play:
        cfg.episode_length_s = 60.0
    return cfg
