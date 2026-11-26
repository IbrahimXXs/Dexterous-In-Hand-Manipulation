"""C5-HW-V2 (experiments/c5_hw_v2/PROTOCOL.md): C5-HW-EFFORT plus the step-A response and the step-B resets.

Base: ``c5hw_effort_env_cfg()`` (GRV2 C5 + the 350 mA-equivalent effort range + the actuator utilisation
penalty, w = 0.2201, unchanged).  Each further change is its own function so it can be tested and ablated alone:

1. ``apply_response``: the action-delay, motor-damping and friction-loss randomisation ranges fitted in
   step A (experiments/c5_hw_v2/results/response_fit.json, ``fitted_randomisation``), frozen here as
   ``RESPONSE``.  ``None`` = the fit failed its pre-registered rule and the change is not applied.
2. ``apply_kit_start_resets``: every episode starts from the kit-start cache of step B
   (experiments/c5_hw_v2/cache/kit_start_cache.npz: the kit's 5-s closing ramp on a cube placed per
   docs/figures/cube_placement.png +- 4 mm / 15 deg, at the deployed effort limit; stored joint target =
   measured post-ramp pose, as the kit resets it).

Nothing else changes: observations, actions, rewards (incl. yaw clip +-0.5 rad/s), terminations, curriculum,
physics and all other randomisation are C5-HW-EFFORT's.
"""

from __future__ import annotations

from pathlib import Path

from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_c5_hw_v1.env_cfgs import c5hw_effort_env_cfg

_PROJECT = Path(__file__).resolve().parents[6]
EXPERIMENT_DIR = _PROJECT / "experiments" / "c5_hw_v2"
KIT_START_CACHE = EXPERIMENT_DIR / "cache" / "kit_start_cache.npz"

# Frozen from experiments/c5_hw_v2/results/response_fit.json before training (PROTOCOL.md); a test checks
# that these equal the results file.  Units: lag_range in physics steps of 5 ms; damping / friction ranges
# are scale factors on the model's dof_damping / dof_frictionloss (as the task's own events use them).
RESPONSE: dict | None = None


def apply_response(cfg, response: dict | None = None):
  r = RESPONSE if response is None else response
  if r is None:
    return cfg
  cfg.events["dr_action_delay"].params["lag_range"] = tuple(int(x) for x in r["lag_range"])
  cfg.events["dr_motor_damping"].params["ranges"] = tuple(float(x) for x in r["damping"])
  cfg.events["dr_motor_friction"].params["ranges"] = tuple(float(x) for x in r["friction"])
  return cfg


def apply_kit_start_resets(cfg, cache: Path | str = KIT_START_CACHE):
  cfg.events["reset_from_grasp_cache"].params["cache_file"] = str(cache)
  cfg.actions["joint_pos"].required_grasp_cache = str(cache)
  return cfg


def c5hw2_env_cfg(play: bool = False):
  return apply_kit_start_resets(apply_response(c5hw_effort_env_cfg(play=play)))
