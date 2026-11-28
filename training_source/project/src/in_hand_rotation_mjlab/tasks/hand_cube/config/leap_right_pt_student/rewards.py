"""Reward term of the optional PT-student action-rate variant (experiments/pt_student/PLAN.md section 4.3)."""

from __future__ import annotations

import torch

ACTION_RATE_WEIGHT = -0.075  # the paper's weight (Yao et al., arXiv 2605.21330 v1); mjlab multiplies every term by dt


def squashed_action_rate_l2(env) -> torch.Tensor:
  """Sum over joints of (tanh u_t - tanh u_{t-1})^2: the change in the executed (squashed) action.

  ``prev_action`` is zero after a reset (mjlab), so the first step of an episode is charged against a zero action."""
  am = env.action_manager
  return torch.sum(torch.square(torch.tanh(am.action) - torch.tanh(am.prev_action)), dim=-1)
