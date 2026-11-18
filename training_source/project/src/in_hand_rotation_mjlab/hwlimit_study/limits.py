"""Approved LEAP Hand joint limits, as enforced by the deployment kit.

Source of truth: ``LEAP_Hand_RL_Deploy_Kit/kit/deploy_direct.py``
(``_HW_LIMIT_LO`` / ``_HW_LIMIT_HI``, "Physical right-hand limits from
leap_hand_controller") with the kit's default ``--limit-margin-rad 0.01``.
The kit clips every command sent to the motors to
``[_HW_LIMIT_LO + margin, _HW_LIMIT_HI - margin]`` (hardware order).

The values are copied here (the kit is not importable from the training
package) and ``tests/test_hwlimit_study.py`` re-parses the kit file to check
that they are identical.  Hardware order is
``[if_rot, if_mcp, if_pip, if_dip, mf_..., rf_..., th_cmc, th_axl, th_mcp, th_ipl]``;
the kit's server/simulation order swaps rot and mcp for the three fingers
(``_HW_TO_SERVER``), giving ``layout.JOINT_NAMES``.
"""

from __future__ import annotations

import numpy as np

from in_hand_rotation_mjlab.estimation.layout import JOINT_NAMES

KIT_HW_LIMIT_LO = (
  -0.523599, -0.261799, -0.261799, -0.261799,
  -0.523599, -0.261799, -0.261799, -0.261799,
  -0.523599, -0.261799, -0.261799, -0.261799,
  0.0, 0.0, -0.349066, -0.349066,
)
KIT_HW_LIMIT_HI = (
  0.523599, 1.91986, 1.91986, 1.91986,
  0.523599, 1.91986, 1.91986, 1.91986,
  0.523599, 1.91986, 1.91986, 1.91986,
  1.571, 1.571, 1.74533, 1.74533,
)
KIT_HW_TO_SERVER = (1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15)
KIT_LIMIT_MARGIN_RAD = 0.01   # deploy_direct.py --limit-margin-rad default
KIT_MAX_COMMAND_STEP_RAD = 0.05  # hand.env MAX_COMMAND_STEP_RAD (powered run aborts above)


def kit_limits_sim_order(margin: float = KIT_LIMIT_MARGIN_RAD) -> tuple[np.ndarray, np.ndarray]:
  """Kit command limits in simulation joint order (``layout.JOINT_NAMES``)."""
  lo = np.asarray(KIT_HW_LIMIT_LO, np.float64)[list(KIT_HW_TO_SERVER)] + margin
  hi = np.asarray(KIT_HW_LIMIT_HI, np.float64)[list(KIT_HW_TO_SERVER)] - margin
  # _HW_TO_SERVER is its own inverse (pairwise swaps), so indexing by it maps
  # hardware order -> server order.
  return lo, hi


def approved_command_limits(soft_lo: np.ndarray, soft_hi: np.ndarray,
                            margin: float = KIT_LIMIT_MARGIN_RAD) -> tuple[np.ndarray, np.ndarray]:
  """Command limits used in training: kit limits intersected with the model's soft limits."""
  klo, khi = kit_limits_sim_order(margin)
  lo, hi = np.maximum(klo, soft_lo), np.minimum(khi, soft_hi)
  if np.any(lo >= hi):
    raise ValueError("empty joint range after intersecting kit and model limits")
  return lo, hi


__all__ = ["JOINT_NAMES", "KIT_HW_LIMIT_LO", "KIT_HW_LIMIT_HI", "KIT_HW_TO_SERVER", "KIT_LIMIT_MARGIN_RAD",
           "KIT_MAX_COMMAND_STEP_RAD", "kit_limits_sim_order", "approved_command_limits"]
