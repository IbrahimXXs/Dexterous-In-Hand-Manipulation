"""Actor models for Experiments C (estimated state) and D (oracle state).

Both are RSL-RL ``MLPModel`` subclasses whose latent is

    [ EmpiricalNormalization(obs_320) | cube_state_features (19) ]

* The 320-D input and its running normaliser are exactly those of the
  baseline actor (Experiment A).
* C computes the features from the frozen estimator applied to obs_320 only.
* D computes them from the ``cube_state`` observation group (MuJoCo ground
  truth; simulation-only, not deployable without external object sensing).
* Both use ``features.cube_state_features`` (fixed scales, canonical cube
  rotation), so C and D receive identical variables, units and normalisation.

``feature_override`` ("zero" | "mean" | "shuffle") is an evaluation-time
ablation switch; it is never set during training.
"""

from __future__ import annotations

from typing import Any

import torch
from rsl_rl.models.mlp_model import MLPModel

from in_hand_rotation_mjlab.estimation.model import load_estimator

from .features import FEATURE_DIM, features_from_estimator, features_from_raw_state

CUBE_STATE_GROUP = "cube_state"


class _FeatureAugmentedMLPModel(MLPModel):
  def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs) -> None:
    object.__setattr__(self, "_feature_dim", FEATURE_DIM)
    super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
    self.feature_override: str | None = None
    self.feature_mean: torch.Tensor | None = None
    self.last_features: torch.Tensor | None = None

  def _get_latent_dim(self) -> int:
    return self.obs_dim + self._feature_dim

  def _raw_obs(self, obs) -> torch.Tensor:
    return torch.cat([obs[g] for g in self.obs_groups], dim=-1)

  def cube_features(self, obs, raw: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
    raise NotImplementedError

  def _apply_override(self, f: torch.Tensor) -> torch.Tensor:
    if self.feature_override is None:
      return f
    if self.feature_override == "zero":
      return torch.zeros_like(f)
    if self.feature_override == "mean":
      if self.feature_mean is None:
        raise RuntimeError("feature_mean not set")
      return self.feature_mean.to(f).expand_as(f)
    if self.feature_override == "shuffle":
      n = f.shape[0]
      return f[(torch.arange(n, device=f.device) + n // 2) % n]
    raise ValueError(self.feature_override)

  def get_latent(self, obs, masks=None, hidden_state=None) -> torch.Tensor:
    raw = self._raw_obs(obs)
    with torch.no_grad():
      f = self.cube_features(obs, raw)
    self.last_features = f
    return torch.cat([self.obs_normalizer(raw), self._apply_override(f)], dim=-1)


class EstimatedStateMLPModel(_FeatureAugmentedMLPModel):
  """Experiment C: features from a frozen estimator applied to obs_320."""

  def __init__(self, obs, obs_groups, obs_set, output_dim, *, estimator_path: str | None = None,
               estimator_state: dict[str, Any] | None = None, **kwargs) -> None:
    super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
    if not estimator_path:
      raise ValueError("EstimatedStateMLPModel needs estimator_path")
    est = load_estimator(estimator_path, map_location="cpu")
    if est.cfg.obs_dim != self.obs_dim:
      raise ValueError(f"estimator input {est.cfg.obs_dim} != actor obs {self.obs_dim}")
    for p in est.parameters():
      p.requires_grad_(False)
    # The frozen estimator is deliberately NOT registered as a submodule.
    # A registered estimator becomes the last nn.Linear in ``self.modules()``,
    # and ``scripts/train.py --zero-actor-output`` zero-initialises exactly that
    # layer: it would silently zero the estimator head (constant features) and
    # leave the actor's own output layer at its default init.  Keeping it out of
    # the module tree also keeps it out of policy checkpoints, the optimiser and
    # ``.train()``; it is pinned to eval mode and moved to the input's device on use.
    object.__setattr__(self, "estimator", est.eval())

  def _estimator_on(self, device: torch.device):
    est = self.estimator
    if next(est.parameters()).device != device:
      est.to(device)
    return est.eval()

  def cube_features(self, obs, raw):
    return features_from_estimator(self._estimator_on(raw.device), raw)


class OracleStateMLPModel(_FeatureAugmentedMLPModel):
  """Experiment D: features from simulator ground truth (``cube_state`` group)."""

  def cube_features(self, obs, raw):
    return features_from_raw_state(obs[CUBE_STATE_GROUP])
