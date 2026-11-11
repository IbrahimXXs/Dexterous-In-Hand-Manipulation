"""
LEAP Hand (Right) constants and EntityCfg for mjlab.

Goals:
- Spawn LEAP RH stably
- Use mjlab IdealPd actuators with current-mapped torque limits
- Embed mesh assets for MuJoCo Warp + Viser
- No task / policy logic
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from math import pi

import mujoco

from in_hand_rotation_mjlab import MYMJLAB_SRC_PATH
from mjlab.actuator import DelayedActuatorCfg, IdealPdActuatorCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

# Optional experiment profile. When unset, the consolidated per-joint SysID
# model remains active. Select H1/H2 before starting the Python process.
LEAP_RIGHT_GAIN_PROFILE_ENV = "LEAP_RIGHT_GAIN_PROFILE"
LEAP_RIGHT_GAIN_PROFILE_DIR = (
    MYMJLAB_SRC_PATH.parent.parent / "configs" / "leap_right_gain_profiles"
)


def _load_gain_profile_from_env() -> dict[str, object] | None:
    requested = os.environ.get(LEAP_RIGHT_GAIN_PROFILE_ENV)
    if requested is None or not requested.strip():
        return None
    name = requested.strip().upper()
    if name not in {"H1", "H2"}:
        raise ValueError(f"{LEAP_RIGHT_GAIN_PROFILE_ENV} must be H1 or H2.")
    path = LEAP_RIGHT_GAIN_PROFILE_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))

LEAP_RIGHT_GAIN_PROFILE = _load_gain_profile_from_env()

# -----------------------------------------------------------------------------
# Actuator SysID constants (DYNAMIXEL XC330-M288)
# -----------------------------------------------------------------------------
#
# We use an ideal PD actuator model with current-domain interpretation:
#   i_des = Kp_i * (q_des - q) - Kd_i * qdot
#   tau_des = K_t * i_des
#
# Datasheet source:
# https://emanual.robotis.com/docs/en/dxl/x/xc330-m288/
#
# Note:
# - We treat Current Limit value as mA (e.g., 550 -> 0.55 A).
# - MuJoCo actuator gains are configured in torque domain:
#     Kp_tau = K_t * Kp_i
#     Kd_tau = K_t * Kd_i
#
LEAP_MOTOR_TORQUE_PER_AMP_NM = 0.53
LEAP_MOTOR_CURRENT_LIMIT_MA = 550.0
LEAP_MOTOR_CURRENT_LIMIT_A = LEAP_MOTOR_CURRENT_LIMIT_MA / 1000.0
LEAP_ACTUATOR_EFFORT_LIMIT_NM = (
    LEAP_MOTOR_TORQUE_PER_AMP_NM * LEAP_MOTOR_CURRENT_LIMIT_A
)

# Current-domain PD gains requested:
#   mI = 800 * joint_error + 200 * d_joint_error
# where mI is in mA, joint_error is in rad, d_joint_error is in rad/s.
LEAP_PD_CURRENT_KP_MA_PER_RAD = 800.0
LEAP_PD_CURRENT_KD_MA_PER_RAD_S = 200.0
LEAP_PD_CURRENT_KP_A_PER_RAD = LEAP_PD_CURRENT_KP_MA_PER_RAD / 1000.0
LEAP_PD_CURRENT_KD_A_PER_RAD_S = LEAP_PD_CURRENT_KD_MA_PER_RAD_S / 1000.0

LEAP_PD_STIFFNESS_NM_PER_RAD = (
    LEAP_MOTOR_TORQUE_PER_AMP_NM * LEAP_PD_CURRENT_KP_A_PER_RAD
)
LEAP_PD_DAMPING_NM_PER_RAD_S = (
    LEAP_MOTOR_TORQUE_PER_AMP_NM * LEAP_PD_CURRENT_KD_A_PER_RAD_S
)

# Match hardware MCP-side behavior in this sim joint convention:
# apply reduced gains on rotational MCP joints {if_rot, mf_rot, rf_rot}.
LEAP_MCP_SIDE_GAIN_SCALE = 0.75
LEAP_MCP_SIDE_PD_STIFFNESS_NM_PER_RAD = (
    LEAP_PD_STIFFNESS_NM_PER_RAD * LEAP_MCP_SIDE_GAIN_SCALE
)
LEAP_MCP_SIDE_PD_DAMPING_NM_PER_RAD_S = (
    LEAP_PD_DAMPING_NM_PER_RAD_S * LEAP_MCP_SIDE_GAIN_SCALE
)

# Reflected armature estimate from LEAP hand SysID note:
# Ia = kg^2 * Ir, with kg=288.35, Ir=1.7e-8 kg m^2.
LEAP_GEAR_RATIO = 288.35
LEAP_ROTOR_INERTIA = 1.7e-8
LEAP_REFLECTED_ARMATURE = (LEAP_GEAR_RATIO**2) * LEAP_ROTOR_INERTIA

# Nominal dry-friction estimate: 10% of max torque (paper suggestion).
LEAP_NOMINAL_FRICTIONLOSS_NM = 0.1 * LEAP_ACTUATOR_EFFORT_LIMIT_NM

# Delay wrapper bounds (in physics steps).
# The env reset event sets the actual lag each episode (e.g., 5..15 for
# decimation=10); keep bounds wide enough to avoid clamping.
LEAP_ACTION_DELAY_MIN_LAG = 0
LEAP_ACTION_DELAY_MAX_LAG = 20
LEAP_ACTION_DELAY_HOLD_PROB = 1.0
LEAP_ACTION_DELAY_UPDATE_PERIOD = 0
LEAP_ACTION_DELAY_PER_ENV_PHASE = True


def configure_leap_spec_for_ideal_pd(spec: mujoco.MjSpec) -> None:
    """Remove XML actuators so mjlab IdealPd actuators are the only actuators."""
    for actuator in tuple(spec.actuators):
        spec.delete(actuator)


# -----------------------------------------------------------------------------
# MJCF + assets
# -----------------------------------------------------------------------------

leap_hand_XML: Path = (
    MYMJLAB_SRC_PATH / "robots" / "leap_hand" / "xmls" / "right_hand.xml"
)
assert leap_hand_XML.exists(), f"Missing MJCF: {leap_hand_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
    """
    Embed mesh assets into MjSpec.assets.

    Required for:
    - mujoco-warp
    - viser
    - distributed viewers
    """
    assets: dict[str, bytes] = {}
    update_assets(
        assets,
        leap_hand_XML.parent.parent / "assets",
        meshdir,
    )
    return assets


def get_spec() -> mujoco.MjSpec:
    """Create MjSpec and attach embedded assets."""
    spec = mujoco.MjSpec.from_file(str(leap_hand_XML))
    configure_leap_spec_for_ideal_pd(spec)
    if LEAP_RIGHT_GAIN_PROFILE is not None:
        mujoco_cfg = LEAP_RIGHT_GAIN_PROFILE["mujoco"]
        assert isinstance(mujoco_cfg, dict)
        passive_damping = float(mujoco_cfg["passive_damping_Nm_s_per_rad"])
        for joint in spec.joints:
            joint.damping[0] = passive_damping
    spec.assets = get_assets(spec.meshdir)
    return spec


# -----------------------------------------------------------------------------
# Joint naming (16 DoF)
# -----------------------------------------------------------------------------
# Index / Middle / Ring / Thumb
LEAP_JOINTS_EXPR = (
    "if_(mcp|rot|pip|dip)",
    "mf_(mcp|rot|pip|dip)",
    "rf_(mcp|rot|pip|dip)",
    "th_(cmc|axl|mcp|ipl)",
)


# -----------------------------------------------------------------------------
# Actuators (ideal PD, current-mapped)
# Per-joint scales below come from the 2026-08-04 right-hand ROS SysID run:
# - index_sysid_fit.npz  (joints 0..3)
# - middle_sysid_fit.npz (joints 4..7)
# - ring_sysid_fit.npz   (joints 8..11)
# - thumb_sysid_fit.npz  (joints 12..15)
#
# Each artifact was fitted from 5,252 physical samples at 20 Hz using
# 2,048 CEM candidates, then cross-validated on the independent 2026-07-29
# physical logs.  The values here are the consolidated absolute fits expressed
# relative to the nominal motor model (not relative to the older SysID scales
# that were active when the fitting run started).
# -----------------------------------------------------------------------------
LEAP_JOINT_ORDER = (
    "if_mcp",
    "if_rot",
    "if_pip",
    "if_dip",
    "mf_mcp",
    "mf_rot",
    "mf_pip",
    "mf_dip",
    "rf_mcp",
    "rf_rot",
    "rf_pip",
    "rf_dip",
    "th_cmc",
    "th_axl",
    "th_mcp",
    "th_ipl",
)

LEAP_MCP_SIDE_ROT_JOINTS = ("if_rot", "mf_rot", "rf_rot")

LEAP_STIFFNESS_SCALE_BY_JOINT = {
    "if_mcp": 1.79911926,
    "if_rot": 2.32661666,
    "if_pip": 1.58431750,
    "if_dip": 1.35909052,
    "mf_mcp": 1.79230542,
    "mf_rot": 2.45886368,
    "mf_pip": 1.70802210,
    "mf_dip": 1.71580092,
    "rf_mcp": 1.60185503,
    "rf_rot": 2.11761580,
    "rf_pip": 1.95858962,
    "rf_dip": 1.43109033,
    "th_cmc": 1.98744741,
    "th_axl": 1.96782447,
    "th_mcp": 1.90028360,
    "th_ipl": 1.72962325,
}
LEAP_DAMPING_SCALE_BY_JOINT = {
    "if_mcp": 0.99213690,
    "if_rot": 0.75241112,
    "if_pip": 0.67153713,
    "if_dip": 0.88468727,
    "mf_mcp": 1.05528818,
    "mf_rot": 0.79049163,
    "mf_pip": 1.06692814,
    "mf_dip": 0.72638236,
    "rf_mcp": 0.84489294,
    "rf_rot": 0.46484626,
    "rf_pip": 0.97262012,
    "rf_dip": 0.83255170,
    "th_cmc": 0.86695872,
    "th_axl": 0.60231940,
    "th_mcp": 1.23894162,
    "th_ipl": 0.85000943,
}
LEAP_EFFORT_SCALE_BY_JOINT = {
    "if_mcp": 0.80846524,
    "if_rot": 1.20528062,
    "if_pip": 0.61412501,
    "if_dip": 0.63065039,
    "mf_mcp": 0.62286726,
    "mf_rot": 1.10042763,
    "mf_pip": 0.82989827,
    "mf_dip": 0.61412501,
    "rf_mcp": 0.69793237,
    "rf_rot": 0.90507794,
    "rf_pip": 0.66704693,
    "rf_dip": 1.24721044,
    "th_cmc": 0.76025586,
    "th_axl": 0.65550693,
    "th_mcp": 0.64456057,
    "th_ipl": 0.95207655,
}
LEAP_ARMATURE_SCALE_BY_JOINT = {
    "if_mcp": 2.14498427,
    "if_rot": 3.04552530,
    "if_pip": 0.77709983,
    "if_dip": 0.66892393,
    "mf_mcp": 0.79638160,
    "mf_rot": 1.91810045,
    "mf_pip": 1.24099692,
    "mf_dip": 1.19463276,
    "rf_mcp": 0.93025189,
    "rf_rot": 1.71440518,
    "rf_pip": 1.12167350,
    "rf_dip": 0.91904441,
    "th_cmc": 1.15222588,
    "th_axl": 0.80199828,
    "th_mcp": 0.71409414,
    "th_ipl": 1.52255272,
}
LEAP_FRICTION_SCALE_BY_JOINT = {
    "if_mcp": 1.37054024,
    "if_rot": 2.32035810,
    "if_pip": 1.95651625,
    "if_dip": 0.07464584,
    "mf_mcp": 0.35654170,
    "mf_rot": 0.07853096,
    "mf_pip": 0.66447087,
    "mf_dip": 0.82757512,
    "rf_mcp": 1.09568054,
    "rf_rot": 0.40836554,
    "rf_pip": 1.06899710,
    "rf_dip": 0.45463652,
    "th_cmc": 0.42049635,
    "th_axl": 0.24833418,
    "th_mcp": 0.27157533,
    "th_ipl": 1.25862267,
}


def _base_stiffness_for_joint(joint_name: str) -> float:
    if joint_name in LEAP_MCP_SIDE_ROT_JOINTS:
        return LEAP_MCP_SIDE_PD_STIFFNESS_NM_PER_RAD
    return LEAP_PD_STIFFNESS_NM_PER_RAD


def _base_damping_for_joint(joint_name: str) -> float:
    if joint_name in LEAP_MCP_SIDE_ROT_JOINTS:
        return LEAP_MCP_SIDE_PD_DAMPING_NM_PER_RAD_S
    return LEAP_PD_DAMPING_NM_PER_RAD_S


def _scaled(value: float, scale_map: dict[str, float], joint_name: str) -> float:
    return value * scale_map.get(joint_name, 1.0)


def _make_joint_actuator_cfg(joint_name: str) -> DelayedActuatorCfg:
    if LEAP_RIGHT_GAIN_PROFILE is None:
        stiffness = _scaled(
            _base_stiffness_for_joint(joint_name),
            LEAP_STIFFNESS_SCALE_BY_JOINT,
            joint_name,
        )
        damping = _scaled(
            _base_damping_for_joint(joint_name),
            LEAP_DAMPING_SCALE_BY_JOINT,
            joint_name,
        )
        armature = _scaled(
            LEAP_REFLECTED_ARMATURE,
            LEAP_ARMATURE_SCALE_BY_JOINT,
            joint_name,
        )
        frictionloss = _scaled(
            LEAP_NOMINAL_FRICTIONLOSS_NM,
            LEAP_FRICTION_SCALE_BY_JOINT,
            joint_name,
        )
    else:
        mujoco_cfg = LEAP_RIGHT_GAIN_PROFILE["mujoco"]
        assert isinstance(mujoco_cfg, dict)
        stiffness = float(mujoco_cfg["kp_Nm_per_rad"])
        damping = float(mujoco_cfg["kv_Nm_s_per_rad"])
        armature = float(mujoco_cfg["armature_kg_m2"])
        frictionloss = float(mujoco_cfg["frictionloss_Nm"])

    base_cfg = IdealPdActuatorCfg(
        target_names_expr=(joint_name,),
        stiffness=stiffness,
        damping=damping,
        effort_limit=_scaled(
            LEAP_ACTUATOR_EFFORT_LIMIT_NM,
            LEAP_EFFORT_SCALE_BY_JOINT,
            joint_name,
        ),
        armature=armature,
        frictionloss=frictionloss,
    )
    return DelayedActuatorCfg(
        base_cfg=base_cfg,
        delay_target="position",
        delay_min_lag=LEAP_ACTION_DELAY_MIN_LAG,
        delay_max_lag=LEAP_ACTION_DELAY_MAX_LAG,
        delay_hold_prob=LEAP_ACTION_DELAY_HOLD_PROB,
        delay_update_period=LEAP_ACTION_DELAY_UPDATE_PERIOD,
        delay_per_env_phase=LEAP_ACTION_DELAY_PER_ENV_PHASE,
    )


LEAP_ACTUATORS = tuple(_make_joint_actuator_cfg(name) for name in LEAP_JOINT_ORDER)

ARTICULATION = EntityArticulationInfoCfg(
    actuators=LEAP_ACTUATORS,
    soft_joint_pos_limit_factor=0.95,
)


# -----------------------------------------------------------------------------
# Initial pose (open, non-singular)
# -----------------------------------------------------------------------------
# Conservative "open hand" pose:
# - MCP slightly flexed
# - DIP/PIP lightly bent
# - Thumb in neutral opposition
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.1),
    joint_pos={
        # Index
        "if_mcp": 0.3,
        "if_rot": 0.0,
        "if_pip": 0.3,
        "if_dip": 0.2,
        # Middle
        "mf_mcp": 0.3,
        "mf_rot": 0.0,
        "mf_pip": 0.3,
        "mf_dip": 0.2,
        # Ring
        "rf_mcp": 0.3,
        "rf_rot": 0.0,
        "rf_pip": 0.3,
        "rf_dip": 0.2,
        # Thumb
        "th_cmc": 0.4,
        "th_axl": 0.2,
        "th_mcp": 0.3,
        "th_ipl": 0.2,
    },
    joint_vel={".*": 0.0},
)


# -----------------------------------------------------------------------------
# Collision configuration
# -----------------------------------------------------------------------------
LEAP_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision.*", ".*_tip"),

    # Per-finger bitmask channels:
    #   bit-0 (1)  = external objects
    #   bit-1 (2)  = index finger
    #   bit-2 (4)  = middle finger
    #   bit-3 (8)  = ring finger
    #   bit-4 (16) = thumb
    #
    # Same-finger pairs share a contype bit that is absent from their own
    # conaffinity, so (contype & conaffinity) == 0 → no intra-finger collision.
    # Cross-finger pairs have different bits → collision enabled.
    # All geoms accept external contacts via bit-0 in conaffinity.
    contype={
        "palm_collision.*": 1,   # external only
        "if_.*": 2,              # index
        "mf_.*": 4,              # middle
        "rf_.*": 8,              # ring
        "th_.*": 16,             # thumb
    },

    conaffinity={
        "palm_collision.*": 1,   # external only
        "if_.*": 29,             # 1 | 4 | 8 | 16  (ext + mid + ring + thumb)
        "mf_.*": 27,             # 1 | 2 | 8 | 16  (ext + idx + ring + thumb)
        "rf_.*": 23,             # 1 | 2 | 4 | 16  (ext + idx + mid + thumb)
        "th_.*": 15,             # 1 | 2 | 4 | 8   (ext + idx + mid + ring)
    },

    # Fingertips get rich contact, others soft
    condim={
        ".*_tip": 6,
        ".*": 3,
    },

    friction={
        ".*_tip": (0.8, 5e-3, 1e-4),
        ".*": (0.2,),
    },

    solref={
        ".*_tip": (0.01, 1),
        ".*": (0.05, 1),
    },

    priority={
        ".*_tip": 2,
        ".*": 0,
    },

    disable_other_geoms=False,
)


# -----------------------------------------------------------------------------
# Final EntityCfg
# -----------------------------------------------------------------------------

def get_leap_hand_cfg() -> EntityCfg:
    """Return an independent mjlab EntityCfg for the LEAP right hand.

    Task builders intentionally mutate their local configuration. Deep copies
    prevent one registered task from changing every other task's configuration.
    """
    return EntityCfg(
        init_state=deepcopy(HOME_KEYFRAME),
        collisions=(deepcopy(LEAP_COLLISION),),
        spec_fn=get_spec,
        articulation=deepcopy(ARTICULATION),
    )


# -----------------------------------------------------------------------------
# Debug entrypoint (native MuJoCo viewer)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import mujoco.viewer
    from mjlab.entity.entity import Entity

    leap = Entity(get_leap_hand_cfg())
    mujoco.viewer.launch(leap.spec.compile())
