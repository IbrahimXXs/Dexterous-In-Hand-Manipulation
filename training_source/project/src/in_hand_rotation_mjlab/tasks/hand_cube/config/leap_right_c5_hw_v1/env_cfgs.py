"""C5-HW-V1 condition C5-HW-effort (experiments/c5_hw_v1/SPEC.md Section 3).

The Grasp-Rotation-V2 C5 environment (``grv2_env_cfg("C5")``: hardware-limited environment,
239-grasp reset cache, 320-D actor observation, tanh delta-position action of +-1/24 rad,
20 Hz, approved command limits, yaw clip +-0.5 rad/s, Retention-V1 retention terms,
terminations, curriculum, physics and all other randomisation unchanged) with exactly two changes:

1. Effort limit.  The reset-time effort-limit randomisation (``dr_effort_limits``), which in
   C5 draws every actuator's limit independently from 0.53 Nm/A x 0.55 A x U(0.9, 1.1)
   = 0.262-0.321 Nm (overwriting the per-joint scales of the actuator config at every reset),
   draws from 0.53 Nm/A x 0.35 A x U(0.9, 1.1) = 0.167-0.204 Nm instead: the deployed
   350 mA current limit in the simulator's own torque/current mapping, same relative spread.
2. Reward ``actuator_utilisation_penalty`` (``c5_hw_v1.rewards``) with the frozen weight
   ``EFFORT_PENALTY_WEIGHT`` (calibrated once before training, see
   experiments/c5_hw_v1/results/effort_weight_calibration.json).  Constant weight: it is not
   part of the penalty curriculum.
"""

from __future__ import annotations

from mjlab.managers.reward_manager import RewardTermCfg

from in_hand_rotation_mjlab.c5_hw_v1.rewards import actuator_utilisation_penalty
from in_hand_rotation_mjlab.robots.leap_hand.leap_right_constants import (
  LEAP_ACTUATOR_EFFORT_LIMIT_NM,
  LEAP_MOTOR_TORQUE_PER_AMP_NM,
)
from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_grasp_rotation_v2.env_cfgs import grv2_env_cfg

DEPLOYED_CURRENT_LIMIT_A = 0.350  # LEAP_Hand_RL_Deploy_Kit/hand.env: current limit 350 mA
EFFORT_PENALTY_WEIGHT: float | None = 0.2201  # frozen 2026-11-24 by experiments/c5_hw_v1/scripts/calibrate_effort_weight.py (w = 0.22006), before training


def deployed_effort_range(base_range: tuple[float, float]) -> tuple[float, float]:
  """The C5 randomisation range rescaled from the 550 mA model limit to the deployed 350 mA."""
  nominal = LEAP_MOTOR_TORQUE_PER_AMP_NM * DEPLOYED_CURRENT_LIMIT_A
  lo, hi = base_range
  return (lo / LEAP_ACTUATOR_EFFORT_LIMIT_NM * nominal, hi / LEAP_ACTUATOR_EFFORT_LIMIT_NM * nominal)


def c5hw_effort_env_cfg(play: bool = False, weight: float | None = None):
  w = EFFORT_PENALTY_WEIGHT if weight is None else weight
  if w is None:
    raise RuntimeError("EFFORT_PENALTY_WEIGHT is not calibrated yet")
  cfg = grv2_env_cfg("C5", play=play)
  ev = cfg.events["dr_effort_limits"]
  ev.params["effort_limit"] = deployed_effort_range(tuple(ev.params["effort_limit"]))
  cfg.rewards["actuator_utilisation_penalty"] = RewardTermCfg(
    func=actuator_utilisation_penalty, weight=-abs(float(w)), params={"asset_name": "robot"})
  return cfg
