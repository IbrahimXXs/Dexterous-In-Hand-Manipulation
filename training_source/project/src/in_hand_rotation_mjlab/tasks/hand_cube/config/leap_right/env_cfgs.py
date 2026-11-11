"""LEAP Right hand cube rotation environment configuration."""

from in_hand_rotation_mjlab.robots import get_leap_right_hand_cfg
from in_hand_rotation_mjlab.tasks.hand_cube.hand_cube_env_cfg import (
    make_hand_cube_embodiment_env_cfg,
)
from in_hand_rotation_mjlab.tasks.hand_cube import mdp as hand_cube_mdp
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg


# LEAP Right specific parameters (mirror of left for now)
GRASP_INIT_JOINT_POS = {
    "if_mcp": 0.1,
    "if_rot": -0.4,  # Mirrored
    "if_pip": 1.3,
    "if_dip": 0.0,
    "mf_mcp": 0.1,
    "mf_rot": 0.0,
    "mf_pip": 1.3,
    "mf_dip": 0.0,
    "rf_mcp": 0.1,
    "rf_rot": 0.4,  # Mirrored
    "rf_pip": 1.3,
    "rf_dip": 0.0,
    "th_cmc": 1.45,
    "th_axl": 1.5,  # Mirrored
    "th_mcp": 0.579,
    "th_ipl": 1.37,
}

CUBE_SPAWN_POS = (-0.01, 0.035, 0.29)  # Modified
GRASP_CACHE_FILE = "tasks/hand_cube/cache/leap_right_grasp_cache.npz"


def leap_right_hand_cube_rotate_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create LEAP Right hand cube rotation environment configuration."""
    cfg = make_hand_cube_embodiment_env_cfg(
        robot_cfg_fn=get_leap_right_hand_cfg,
        grasp_init_joint_pos=GRASP_INIT_JOINT_POS,
        cube_spawn_pos=CUBE_SPAWN_POS,
        grasp_cache_file=GRASP_CACHE_FILE,
        negate_yaw_rate=False,  # Right hand: counter-clockwise rotation is positive
        play=play,
    )
    cfg.rewards["fingertip_participation"] = RewardTermCfg(
        func=hand_cube_mdp.fingertip_contact_participation,
        weight=0.15,
        params={
            "sensor_name": "fingertip_cube_contact",
            # Two seconds at the 20 Hz policy rate. This rewards a full gait
            # cycle without forcing simultaneous contact.
            "window_steps": 40,
            "expected_fingertips": 4,
            # Three participating tips score 0.75**3 = 0.422, while all four
            # score 1.0. This makes the least-used fingertip materially useful.
            "participation_power": 3.0,
        },
    )
    cfg.rewards["fingertip_proximity"] = RewardTermCfg(
        func=hand_cube_mdp.fingertip_cube_proximity,
        weight=0.12,
        params={
            "hand_cfg": SceneEntityCfg(
                "robot",
                geom_names=("if_tip", "mf_tip", "rf_tip", "th_tip"),
                preserve_order=True,
            ),
            "cube_cfg": SceneEntityCfg(
                "cube",
                geom_names=("cube_geom",),
                preserve_order=True,
            ),
            # Keep each fingertip near enough to remain visually and
            # mechanically engaged while still permitting a release/regrasp.
            "surface_margin": 0.018,
            "distance_scale": 0.025,
            # The least-engaged fingertip matters more than the easy average.
            "worst_tip_weight": 0.6,
        },
    )
    cfg.rewards["instant_fingertip_contact"] = RewardTermCfg(
        func=hand_cube_mdp.fingertip_cube_contact_fraction,
        weight=0.04,
        params={
            "sensor_name": "fingertip_cube_contact",
            "max_contacts": 4,
        },
    )
    cfg.rewards["non_thumb_curl"] = RewardTermCfg(
        func=hand_cube_mdp.non_thumb_finger_curl,
        weight=0.03,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=("if_pip", "mf_pip", "rf_pip"),
                preserve_order=True,
            ),
            "open_angle": 0.45,
            "engaged_angle": 0.90,
            "worst_finger_weight": 0.5,
        },
    )
    return cfg
