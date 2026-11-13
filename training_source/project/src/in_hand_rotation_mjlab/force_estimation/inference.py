"""Standalone inference for the fingertip force estimator (no simulator needed).

Input (identical to the policy / cube-state estimator, ``estimation/layout.py``)::

    obs[0:160]   joint_pos_rel        10 frames x 16 joints, oldest -> newest
                                      (measured joint position - default pose, rad)
    obs[160:320] prev_commanded_joint_pos  10 frames x 16 joints, oldest -> newest
                                      (absolute commanded target, rad)
    joint order: estimation.layout.JOINT_NAMES

Output: ``[thumb_force_N, index_force_N, middle_force_N, ring_force_N]`` -
compressive normal force between each fingertip (distal link) and the object,
newtons, >= 0.  Normalisation is inside the checkpoint; pass raw observations.

Example::

    est = ForceEstimatorRuntime("experiments/force_estimation/models/force_estimator.pt")
    forces = est.predict(obs320)          # np.ndarray shape (4,) or (B, 4)
"""

from __future__ import annotations

import numpy as np
import torch

from .constants import FORCE_FINGER_NAMES
from .model import load_force_estimator


class ForceEstimatorRuntime:
  output_names = tuple(f"{n}_force_N" for n in FORCE_FINGER_NAMES)

  def __init__(self, checkpoint: str, device: str = "cpu"):
    self.device = torch.device(device)
    self.model = load_force_estimator(checkpoint, map_location=self.device).to(self.device).eval()
    for p in self.model.parameters():
      p.requires_grad_(False)

  @torch.inference_mode()
  def predict(self, obs) -> np.ndarray:
    x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
    single = x.ndim == 1
    if single:
      x = x[None]
    if x.shape[-1] != self.model.cfg.obs_dim or x.ndim != 2:
      raise ValueError(f"expected (320,) or (B, 320), got {tuple(x.shape)}")
    if not torch.isfinite(x).all():
      raise ValueError("non-finite observation")
    y = self.model(x).cpu().numpy()
    return y[0] if single else y

  @torch.inference_mode()
  def contact_probability(self, obs) -> np.ndarray | None:
    x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
    p = self.model.contact_probability(x if x.ndim == 2 else x[None])
    if p is None:
      return None
    p = p.cpu().numpy()
    return p if np.asarray(obs).ndim == 2 else p[0]
