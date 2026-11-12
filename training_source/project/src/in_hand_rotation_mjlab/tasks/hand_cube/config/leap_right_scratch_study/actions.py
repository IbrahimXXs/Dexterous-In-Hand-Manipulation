"""Squashed delta-position action for the from-scratch study (see README)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from in_hand_rotation_mjlab.tasks.hand_cube.mdp.actions import (
  JointPositionDeltaAction,
  JointPositionDeltaActionCfg,
)


# --------------------------------------------------------------------------
@dataclass(kw_only=True)
class SquashedJointPositionDeltaActionCfg(JointPositionDeltaActionCfg):
  """Original delta-position action; the policy output u is mapped to the
  normalised command by ``tanh`` ("tanh") or by clipping ("clip", original).

  With "tanh" the action the environment executes is a smooth bijection of
  the Gaussian sample PPO stores, so the PPO likelihood ratio is exact for the
  executed action (the tanh Jacobian cancels) and there is no region of
  zero sensitivity.  The physical per-step delta range is unchanged.
  """

  squash: str = "tanh"

  def build(self, env):
    return SquashedJointPositionDeltaAction(self, env)


class SquashedJointPositionDeltaAction(JointPositionDeltaAction):
  cfg: SquashedJointPositionDeltaActionCfg

  def process_actions(self, actions: torch.Tensor):
    if self.cfg.squash == "tanh":
      normalized = torch.tanh(actions)
    elif self.cfg.squash == "clip":
      normalized = actions.clamp(-1.0, 1.0)
    else:
      raise ValueError(f"unknown squash {self.cfg.squash!r}")
    self.executed_normalized = normalized
    super().process_actions(normalized)
