"""C5-HW-V1 reward term (simulator state only; nothing here reaches the actor).

``actuator_utilisation_penalty`` (experiments/c5_hw_v1/SPEC.md, condition C5-HW-effort):

    u_j = |tau_j| / L_j                      tau_j: actuator force, L_j: that environment's
                                             current effort limit of actuator j (forcerange)
    p   = sum_j [ max(0, u_j - 0.8) / 0.2 ]^2

p is 0 while every actuator is below 80% of its limit, rises smoothly (C1) above it and
reaches 1 per actuator at the limit.  The reward contribution is -w * p (w = the term weight).
"""

from __future__ import annotations

import torch

UTIL_THRESHOLD = 0.8


def utilisation_hinge(u: torch.Tensor, threshold: float = UTIL_THRESHOLD) -> torch.Tensor:
  """sum_j [max(0, u_j - threshold) / (1 - threshold)]^2 over the last axis."""
  return torch.square(torch.clamp(u - threshold, min=0.0) / (1.0 - threshold)).sum(-1)


def actuator_utilisation(env, asset_name: str = "robot") -> torch.Tensor:
  """u (num_envs, num_actuators): |actuator force| / the environment's effort limit (same ctrl order)."""
  asset = env.scene[asset_name]
  ctrl = asset.indexing.ctrl_ids
  fr = env.sim.model.actuator_forcerange
  lim = fr[:, ctrl, 1] if fr.ndim == 3 else fr[ctrl, 1].unsqueeze(0)
  return asset.data.actuator_force.abs() / torch.clamp(lim, min=1e-9)


def actuator_utilisation_penalty(env, asset_name: str = "robot", threshold: float = UTIL_THRESHOLD) -> torch.Tensor:
  return utilisation_hinge(actuator_utilisation(env, asset_name), threshold)
