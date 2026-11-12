"""RSL-RL PPO with additional diagnostics (the optimisation itself is unchanged).

Logged every iteration under ``Loss/`` in TensorBoard:

  approx_kl              mean KL(old || new) over 16k rollout samples after the update
  grad_norm_actor        mean pre-clipping actor gradient norm over minibatches
  grad_norm_critic       same for the critic
  raw_action_abs_mean    mean |u| of sampled pre-transform actions
  raw_mean_abs_mean      mean |mu|
  exec_saturation        share of executed (transformed) entries with |a| > 0.99
  exec_mean_saturation   same for the deterministic mean action
  effective_exploration  share of entries where |T(u) - T(mu)| > 0.05 (normalised
                         command units), i.e. exploration changes what is executed
  effective_exploration_001  same with threshold 0.01
  action_std_mean        mean policy standard deviation

``action_transform`` must match the environment's action squashing
("tanh" or "clip").
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO


class InstrumentedPPO(PPO):
  def __init__(self, *args, action_transform: str = "tanh", **kwargs):
    super().__init__(*args, **kwargs)
    if action_transform not in ("tanh", "clip"):
      raise ValueError(action_transform)
    self.action_transform = action_transform

  def _T(self, x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x) if self.action_transform == "tanh" else x.clamp(-1.0, 1.0)

  def update(self) -> dict[str, float]:
    st = self.storage
    with torch.no_grad():
      u = st.actions.flatten(0, 1)
      mu = st.mu.flatten(0, 1)
      sigma = st.sigma.flatten(0, 1)
      tu, tmu = self._T(u), self._T(mu)
      d = (tu - tmu).abs()
      stats = {
        "raw_action_abs_mean": float(u.abs().mean()),
        "raw_mean_abs_mean": float(mu.abs().mean()),
        "exec_saturation": float((tu.abs() > 0.99).float().mean()),
        "exec_mean_saturation": float((tmu.abs() > 0.99).float().mean()),
        "effective_exploration": float((d > 0.05).float().mean()),
        "effective_exploration_001": float((d > 0.01).float().mean()),
        "action_std_mean": float(sigma.mean()),
      }
      n = u.shape[0]
      idx = torch.randperm(n, device=u.device)[: min(n, 16384)]
      obs_sub = st.observations.flatten(0, 1)[idx]
      old_mu, old_sigma = mu[idx].clone(), sigma[idx].clone()

    norms: list[float] = []
    orig = nn.utils.clip_grad_norm_

    def recording(params, max_norm, *a, **k):
      total = orig(params, max_norm, *a, **k)
      norms.append(float(total))
      return total

    nn.utils.clip_grad_norm_ = recording
    try:
      loss_dict = super().update()
    finally:
      nn.utils.clip_grad_norm_ = orig

    with torch.no_grad():
      self.actor(obs_sub, stochastic_output=True)
      new_mu, new_sigma = self.actor.output_mean, self.actor.output_std
      kl = torch.sum(
        torch.log(new_sigma / old_sigma + 1e-5)
        + (old_sigma.square() + (old_mu - new_mu).square()) / (2.0 * new_sigma.square())
        - 0.5,
        dim=-1,
      )
      stats["approx_kl"] = float(kl.mean())
    if norms:
      stats["grad_norm_actor"] = sum(norms[0::2]) / len(norms[0::2])
      stats["grad_norm_critic"] = sum(norms[1::2]) / max(len(norms[1::2]), 1)
    loss_dict.update(stats)
    return loss_dict
