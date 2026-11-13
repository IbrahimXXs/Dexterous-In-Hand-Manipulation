"""Fingertip normal-force estimator: 320-D proprioceptive history -> 4 forces [N].

Independent of the cube-state estimator B (no shared weights or files).  The
trunk mirrors B's MLP (512-512-256, LayerNorm, ELU) for comparability.

Heads
-----
``direct``  4 outputs -> softplus -> force / scale.  Non-negative, finite.
``hurdle``  4 contact logits + 4 magnitudes (softplus).  Prediction is the
            expected force  p_contact * magnitude  (non-negative, finite);
            ``p_contact`` is also exposed for contact classification.

Input normalisation (training-split mean/std per input dimension) and per-finger
target scales are stored inside the checkpoint as buffers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

from in_hand_rotation_mjlab.estimation.layout import ACTOR_OBS_DIM

from .constants import FORCE_FINGER_NAMES, NUM_FORCE_FINGERS

CHECKPOINT_FORMAT = "leap_fingertip_force_estimator_v1"


@dataclass
class ForceEstimatorConfig:
  obs_dim: int = ACTOR_OBS_DIM
  hidden_dims: tuple[int, ...] = (512, 512, 256)
  layer_norm: bool = True
  head: str = "direct"  # "direct" | "hurdle"
  finger_order: tuple[str, ...] = field(default_factory=lambda: tuple(FORCE_FINGER_NAMES))


class FingertipForceEstimator(nn.Module):
  def __init__(self, cfg: ForceEstimatorConfig):
    super().__init__()
    if cfg.head not in ("direct", "hurdle"):
      raise ValueError(cfg.head)
    self.cfg = cfg
    layers: list[nn.Module] = []
    d = cfg.obs_dim
    for h in cfg.hidden_dims:
      layers.append(nn.Linear(d, h))
      if cfg.layer_norm:
        layers.append(nn.LayerNorm(h))
      layers.append(nn.ELU())
      d = h
    self.trunk = nn.Sequential(*layers)
    n_out = NUM_FORCE_FINGERS * (2 if cfg.head == "hurdle" else 1)
    self.head = nn.Linear(d, n_out)
    self.register_buffer("obs_mean", torch.zeros(cfg.obs_dim))
    self.register_buffer("obs_std", torch.ones(cfg.obs_dim))
    self.register_buffer("force_scale", torch.ones(NUM_FORCE_FINGERS))  # N per normalised unit

  def set_normalization(self, obs_mean, obs_std, force_scale) -> None:
    self.obs_mean.copy_(torch.as_tensor(obs_mean, dtype=torch.float32))
    self.obs_std.copy_(torch.as_tensor(obs_std, dtype=torch.float32).clamp_min(1e-6))
    self.force_scale.copy_(torch.as_tensor(force_scale, dtype=torch.float32).clamp_min(1e-6))

  def raw_outputs(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Normalised-unit outputs used by the loss."""
    if obs.shape[-1] != self.cfg.obs_dim:
      raise ValueError(f"expected [..., {self.cfg.obs_dim}] observation, got {tuple(obs.shape)}")
    x = (obs - self.obs_mean) / self.obs_std
    z = self.head(self.trunk(x))
    if self.cfg.head == "direct":
      return {"force_n": F.softplus(z)}
    logit, mag = z[..., :NUM_FORCE_FINGERS], F.softplus(z[..., NUM_FORCE_FINGERS:])
    return {"logit": logit, "mag_n": mag, "force_n": torch.sigmoid(logit) * mag}

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    """[..., 320] -> [..., 4] non-negative forces in newtons (thumb, index, middle, ring)."""
    return self.raw_outputs(obs)["force_n"] * self.force_scale

  @torch.no_grad()
  def contact_probability(self, obs: torch.Tensor) -> torch.Tensor | None:
    if self.cfg.head != "hurdle":
      return None
    return torch.sigmoid(self.raw_outputs(obs)["logit"])

  def checkpoint(self, extra: dict | None = None) -> dict:
    cfg = asdict(self.cfg)
    return {"format": CHECKPOINT_FORMAT, "config": cfg, "state_dict": self.state_dict(), "extra": extra or {}}

  @classmethod
  def from_checkpoint(cls, ckpt: dict) -> "FingertipForceEstimator":
    if ckpt.get("format") != CHECKPOINT_FORMAT:
      raise ValueError(f"not a fingertip force-estimator checkpoint (format={ckpt.get('format')!r})")
    c = dict(ckpt["config"])
    c["hidden_dims"] = tuple(c["hidden_dims"])
    c["finger_order"] = tuple(c["finger_order"])
    model = cls(ForceEstimatorConfig(**c))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model


def load_force_estimator(path, map_location="cpu") -> FingertipForceEstimator:
  ckpt = torch.load(path, map_location=map_location, weights_only=False)
  return FingertipForceEstimator.from_checkpoint(ckpt).eval()
