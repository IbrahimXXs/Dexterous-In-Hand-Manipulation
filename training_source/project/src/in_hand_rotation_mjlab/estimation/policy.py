"""Estimator-augmented PPO actor for RSL-RL 4.x.

The environment's actor observation is unchanged (320-D, real-robot
available).  Inside the actor::

    obs_320 --(EmpiricalNormalization, as before)--> z_320 ----------+
       |                                                              |-> MLP -> action mean
       +--(frozen CubeStateEstimator)--> features (19 or 4) ----------+

So the actor input becomes ``320 + F`` while the environment, the critic
(83-D privileged), the reward and the action semantics are untouched.  No
ground-truth cube state ever reaches the actor: the only input is ``obs_320``.

Normalisation: the 320 raw dimensions keep RSL-RL's running normaliser
(statistics migrated from model_400).  Estimator features are normalised by
the estimator itself with fixed training-set statistics (positions and
velocities standardised and clamped to +-5; rotation as unit 6D columns;
contacts as probabilities), so they are not fed through the running
normaliser and cannot drift while PPO updates its statistics.

Selected through ``class_name =
"in_hand_rotation_mjlab.estimation.policy:EstimatorAugmentedMLPModel"``.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel

from .layout import FEATURE_MODES
from .model import CubeStateEstimator, EstimatorConfig, load_estimator


class EstimatorAugmentedMLPModel(MLPModel):
  def __init__(
    self,
    obs,
    obs_groups,
    obs_set,
    output_dim,
    *,
    estimator_path: str | None = None,
    estimator_config: dict[str, Any] | None = None,
    feature_mode: str = "full",
    freeze_estimator: bool = True,
    **kwargs,
  ) -> None:
    if feature_mode not in FEATURE_MODES:
      raise ValueError(f"feature_mode must be one of {list(FEATURE_MODES)}")
    # _get_latent_dim() is called inside MLPModel.__init__, so the feature
    # width must be known before the parent constructor runs.
    object.__setattr__(self, "_feature_dim", FEATURE_MODES[feature_mode])
    super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
    self.feature_mode = feature_mode
    self.freeze_estimator = freeze_estimator

    if estimator_path:
      estimator = load_estimator(estimator_path)
    elif estimator_config is not None:
      # Architecture only; weights arrive through load_state_dict
      # (augmented checkpoints contain ``estimator.*`` tensors).
      cfg = dict(estimator_config)
      cfg["hidden_dims"] = tuple(cfg["hidden_dims"])
      estimator = CubeStateEstimator(EstimatorConfig(**cfg))
    else:
      raise ValueError("Provide estimator_path or estimator_config.")
    if estimator.cfg.obs_dim != self.obs_dim:
      raise ValueError(
        f"Estimator input {estimator.cfg.obs_dim} != actor observation {self.obs_dim}"
      )
    self.estimator = estimator
    if freeze_estimator:
      for p in self.estimator.parameters():
        p.requires_grad_(False)
      self.estimator.eval()

  # -- MLPModel hooks --------------------------------------------------------
  def _get_latent_dim(self) -> int:
    return self.obs_dim + self._feature_dim

  def _raw_obs(self, obs) -> torch.Tensor:
    return torch.cat([obs[g] for g in self.obs_groups], dim=-1)

  def estimator_features(self, raw: torch.Tensor) -> torch.Tensor:
    if self.freeze_estimator:
      with torch.no_grad():
        return self.estimator.policy_features(raw, self.feature_mode)
    return self.estimator.policy_features(raw, self.feature_mode)

  def get_latent(self, obs, masks=None, hidden_state=None) -> torch.Tensor:
    raw = self._raw_obs(obs)
    return torch.cat([self.obs_normalizer(raw), self.estimator_features(raw)], dim=-1)

  def train(self, mode: bool = True):
    super().train(mode)
    if self.freeze_estimator:
      self.estimator.eval()
    return self

  # -- export ----------------------------------------------------------------
  def as_jit(self) -> nn.Module:
    return _ExportAugmentedActor(self)

  def as_onnx(self, verbose: bool) -> nn.Module:
    return _ExportAugmentedActor(self, onnx=True)


class _ExportAugmentedActor(nn.Module):
  """Deterministic export: obs_320 -> action mean (JIT / ONNX)."""

  is_recurrent: bool = False

  def __init__(self, model: EstimatorAugmentedMLPModel, onnx: bool = False):
    super().__init__()
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.estimator = copy.deepcopy(model.estimator).eval()
    self.mlp = copy.deepcopy(model.mlp)
    self.feature_mode = model.feature_mode
    self.input_size = model.obs_dim

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    feats = self.estimator.policy_features(x, self.feature_mode)
    return self.mlp(torch.cat([self.obs_normalizer(x), feats], dim=-1))

  def get_dummy_inputs(self):
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self):
    return ["obs"]

  @property
  def output_names(self):
    return ["actions"]

  def reset(self) -> None:
    pass
