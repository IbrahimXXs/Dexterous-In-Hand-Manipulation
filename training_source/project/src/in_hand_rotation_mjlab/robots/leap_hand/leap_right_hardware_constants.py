"""Right LEAP Hand model with deployment-matched joint position limits.

All actuator dynamics, collision settings, assets, and SysID parameters are
inherited from :mod:`leap_right_constants`. Only MjSpec joint ranges differ.
"""

from __future__ import annotations

import mujoco

from .leap_right_constants import (
    get_leap_hand_cfg as _get_base_hand_cfg,
    get_spec as _get_base_spec,
)


# Server/training joint names mapped to the same physical limits enforced by
# real_world/deploy_direct.py. Values are radians and deliberately omit the
# deployment-only 0.01 rad command margin.
LEAP_HARDWARE_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "if_mcp": (-0.261799, 1.91986),
    "if_rot": (-0.523599, 0.523599),
    "if_pip": (-0.261799, 1.91986),
    "if_dip": (-0.261799, 1.91986),
    "mf_mcp": (-0.261799, 1.91986),
    "mf_rot": (-0.523599, 0.523599),
    "mf_pip": (-0.261799, 1.91986),
    "mf_dip": (-0.261799, 1.91986),
    "rf_mcp": (-0.261799, 1.91986),
    "rf_rot": (-0.523599, 0.523599),
    "rf_pip": (-0.261799, 1.91986),
    "rf_dip": (-0.261799, 1.91986),
    "th_cmc": (0.0, 1.571),
    "th_axl": (0.0, 1.571),
    "th_mcp": (-0.349066, 1.74533),
    "th_ipl": (-0.349066, 1.74533),
}


def get_spec() -> mujoco.MjSpec:
    """Return the SysID right-hand spec with hardware-matched joint ranges."""
    spec = _get_base_spec()
    for joint_name, joint_range in LEAP_HARDWARE_JOINT_LIMITS.items():
        joint = spec.joint(joint_name)
        if joint is None:
            raise KeyError(f"Right-hand MjSpec has no joint {joint_name!r}")
        joint.range = joint_range
        joint.limited = True
    return spec


def get_leap_right_hardware_hand_cfg():
    """Return the existing SysID hand configuration with the limited spec."""
    cfg = _get_base_hand_cfg()
    cfg.spec_fn = get_spec
    return cfg
