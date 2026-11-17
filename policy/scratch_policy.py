#!/usr/bin/env python3
"""Robot-side runner for the from-scratch study policies (A and C).

Loads a ``leap_scratch_policy/v1`` TorchScript bundle written by
``training_source/project/scripts/estimation/export_scratch_policy.py``.

Identical robot-side pipeline for A and C (the only difference is inside the
bundle: C's actor reads estimator-B features, A's does not):

* 320-D history (joint_pos_rel + previous absolute commands, 10 frames each)
  maintained exactly as in :class:`policy.yaw_policy.ContinuousYawPolicy`.
* Training normalisation ``(x - mean) / (std + 1e-2)`` (inside the bundle).
* Executed action ``tanh(u)`` (the study's action squash), scaled by 1/24 rad,
  added to the previous command, clipped to joint limits.

After every :meth:`step`, ``self.last_outputs`` holds the monitors computed
from the same observation (both A and C): estimated cube state (palm frame),
the 19 cube features, and fingertip normal forces (thumb, index, middle, ring).

Safety: this class only computes targets.  Physical runs go through
``hardware/deploy_direct.py`` with the supervised procedure in TA_GUIDE.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from policy.yaw_policy import ContinuousYawPolicy

BUNDLE_FORMAT = "leap_scratch_policy/v1"


class ScratchStudyPolicy(ContinuousYawPolicy):
  def __init__(self, bundle_path: str | Path, device: str = "cpu"):
    self._device = device
    self.last_outputs: dict[str, np.ndarray] | None = None
    super().__init__(bundle_path)
    self.condition = self.meta["condition"]

  def _load_weights(self, path: Path) -> None:
    import torch

    torch.set_num_threads(1)  # deterministic latency at 20 Hz; the net is tiny
    extra = {"meta.json": ""}
    self._bundle = torch.jit.load(str(path), map_location=self._device, _extra_files=extra)
    self._bundle.eval()
    self.meta = json.loads(extra["meta.json"] or "{}")
    if self.meta.get("format") != BUNDLE_FORMAT:
      raise ValueError(f"{path} is not a {BUNDLE_FORMAT} bundle")
    if self.meta.get("privileged_inputs", True):
      raise ValueError("Bundle declares privileged inputs; refusing to deploy.")
    if self.meta.get("action_squash") != "tanh":
      raise ValueError(f"unsupported action squash {self.meta.get('action_squash')!r}")
    if abs(float(self.meta.get("delta_range_rad", 0.0)) - self.delta_range) > 1e-12:
      raise ValueError("bundle delta range differs from the runner's")

  def _forward(self, obs_320: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
      x = torch.as_tensor(obs_320, dtype=torch.float32, device=self._device).reshape(1, 320)
      raw, action, state, feats, forces = self._bundle(x)
    self._last_tanh = action[0].cpu().numpy().reshape(-1)
    self.last_outputs = {
      "obs": np.asarray(obs_320, np.float32).reshape(-1).copy(),
      "action": self._last_tanh.copy(),
      "cube_state": state[0].cpu().numpy(),
      "features": feats[0].cpu().numpy(),
      "forces": forces[0].cpu().numpy(),
    }
    return raw[0].cpu().numpy().reshape(-1)

  def _squash_action(self, raw_action: np.ndarray) -> np.ndarray:
    # Use the bundle's own tanh (bit-identical to what the actor was trained on).
    return self._last_tanh.astype(np.float32)
