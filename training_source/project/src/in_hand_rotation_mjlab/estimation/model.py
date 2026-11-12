"""Proprioceptive cube-state estimator (MLP baseline).

Input  : the unchanged 320-D actor observation (joint-position history and
         previous-command history; see ``layout.py``).  Nothing else.
Output : cube position, orientation (6D), linear/angular velocity in the palm
         frame, and four fingertip-contact logits.

Normalisation is part of the module (registered buffers), so a saved
estimator is self-contained: raw observations in, physical units out.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .layout import ACTOR_OBS_DIM, FEATURE_MODES, NUM_FINGERTIPS
from .rotations import matrix_to_quat, rot6d_to_matrix

_OUT_POS = slice(0, 3)
_OUT_ROT = slice(3, 9)
_OUT_LIN = slice(9, 12)
_OUT_ANG = slice(12, 15)
_OUT_CON = slice(15, 15 + NUM_FINGERTIPS)
OUTPUT_DIM = 15 + NUM_FINGERTIPS


@dataclass
class EstimatorConfig:
  obs_dim: int = ACTOR_OBS_DIM
  hidden_dims: tuple[int, ...] = (512, 512, 256)
  activation: str = "elu"
  layer_norm: bool = True
  dropout: float = 0.0
  # Extra metadata stored with checkpoints (dataset path, git rev, ...).
  meta: dict[str, Any] = field(default_factory=dict)


def _activation(name: str) -> nn.Module:
  return {"elu": nn.ELU(), "relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}[
    name
  ]


class CubeStateEstimator(nn.Module):
  def __init__(self, cfg: EstimatorConfig | None = None):
    super().__init__()
    self.cfg = cfg or EstimatorConfig()
    layers: list[nn.Module] = []
    in_dim = self.cfg.obs_dim
    for h in self.cfg.hidden_dims:
      layers.append(nn.Linear(in_dim, h))
      if self.cfg.layer_norm:
        layers.append(nn.LayerNorm(h))
      layers.append(_activation(self.cfg.activation))
      if self.cfg.dropout > 0:
        layers.append(nn.Dropout(self.cfg.dropout))
      in_dim = h
    self.trunk = nn.Sequential(*layers)
    self.head = nn.Linear(in_dim, OUTPUT_DIM)

    # Input normalisation (set from the training split).
    self.register_buffer("obs_mean", torch.zeros(self.cfg.obs_dim))
    self.register_buffer("obs_std", torch.ones(self.cfg.obs_dim))
    # Target normalisation for continuous outputs (set from training split).
    self.register_buffer("pos_mean", torch.zeros(3))
    self.register_buffer("pos_std", torch.ones(3))
    self.register_buffer("lin_vel_std", torch.ones(3))
    self.register_buffer("ang_vel_std", torch.ones(3))

  # ------------------------------------------------------------------ stats
  @torch.no_grad()
  def set_normalization(
    self,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    pos_mean: torch.Tensor,
    pos_std: torch.Tensor,
    lin_vel_std: torch.Tensor,
    ang_vel_std: torch.Tensor,
    min_std: float = 1e-4,
  ) -> None:
    self.obs_mean.copy_(obs_mean)
    self.obs_std.copy_(obs_std.clamp_min(min_std))
    self.pos_mean.copy_(pos_mean)
    self.pos_std.copy_(pos_std.clamp_min(min_std))
    self.lin_vel_std.copy_(lin_vel_std.clamp_min(min_std))
    self.ang_vel_std.copy_(ang_vel_std.clamp_min(min_std))

  # ---------------------------------------------------------------- forward
  def forward_raw(self, obs: torch.Tensor) -> torch.Tensor:
    """Network output in *normalised* units: [N, 19]."""
    if obs.shape[-1] != self.cfg.obs_dim:
      raise ValueError(
        f"Estimator expects {self.cfg.obs_dim}-D actor observations, got "
        f"{obs.shape[-1]}."
      )
    x = (obs - self.obs_mean) / self.obs_std
    return self.head(self.trunk(x))

  def decode(self, raw: torch.Tensor) -> dict[str, torch.Tensor]:
    rot6d = raw[..., _OUT_ROT]
    rotmat = rot6d_to_matrix(rot6d)
    return {
      "pos": raw[..., _OUT_POS] * self.pos_std + self.pos_mean,
      "rot6d": rot6d,
      "rotmat": rotmat,
      "lin_vel": raw[..., _OUT_LIN] * self.lin_vel_std,
      "ang_vel": raw[..., _OUT_ANG] * self.ang_vel_std,
      "contact_logits": raw[..., _OUT_CON],
      "contact_prob": torch.sigmoid(raw[..., _OUT_CON]),
      # normalised continuous outputs (used by the loss)
      "pos_n": raw[..., _OUT_POS],
      "lin_vel_n": raw[..., _OUT_LIN],
      "ang_vel_n": raw[..., _OUT_ANG],
    }

  def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
    return self.decode(self.forward_raw(obs))

  @torch.no_grad()
  def predict(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
    out = self.forward(obs)
    out["quat"] = matrix_to_quat(out["rotmat"])
    return out

  def policy_features(self, obs: torch.Tensor, mode: str = "full") -> torch.Tensor:
    """Bounded / normalised features appended to the actor input.

    ``full``    : pos_n(3) | orthonormal 6D rotation(6) | lin_vel_n(3) |
                  ang_vel_n(3) | contact_prob(4)                   -> 19
    ``contact`` : contact_prob(4)                                  -> 4

    The rotation columns come from the Gram-Schmidt matrix (unit vectors) and
    contact values are probabilities, so these features are already on an
    O(1) scale; continuous quantities use the training-set statistics.  This
    is why the actor's empirical normaliser is *not* applied to them.
    """
    if mode not in FEATURE_MODES:
      raise ValueError(f"Unknown feature mode {mode!r}; options {list(FEATURE_MODES)}")
    out = self.forward(obs)
    if mode == "contact":
      return out["contact_prob"]
    rot = out["rotmat"]
    rot6d_ortho = torch.cat([rot[..., :, 0], rot[..., :, 1]], dim=-1)
    return torch.cat(
      [
        out["pos_n"].clamp(-5.0, 5.0),
        rot6d_ortho,
        out["lin_vel_n"].clamp(-5.0, 5.0),
        out["ang_vel_n"].clamp(-5.0, 5.0),
        out["contact_prob"],
      ],
      dim=-1,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_estimator(model: CubeStateEstimator, path: str | Path, **extra: Any) -> None:
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  cfg = asdict(model.cfg)
  cfg["hidden_dims"] = list(cfg["hidden_dims"])
  torch.save(
    {
      "format": "cube_state_estimator/v1",
      "config": cfg,
      "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
      **extra,
    },
    path,
  )


def load_estimator(
  path: str | Path, map_location: str | torch.device = "cpu"
) -> CubeStateEstimator:
  blob = torch.load(Path(path), map_location=map_location, weights_only=False)
  if blob.get("format") != "cube_state_estimator/v1":
    raise ValueError(f"{path} is not a cube_state_estimator/v1 checkpoint")
  cfg_dict = dict(blob["config"])
  cfg_dict["hidden_dims"] = tuple(cfg_dict["hidden_dims"])
  model = CubeStateEstimator(EstimatorConfig(**cfg_dict))
  model.load_state_dict(blob["state_dict"], strict=True)
  return model.to(map_location)
