#!/usr/bin/env python3
"""Robot-side runner for the estimator-augmented continuous-yaw policy.

Drop-in replacement for :class:`policy.yaw_policy.ContinuousYawPolicy` that
loads a TorchScript bundle written by
``training_source/project/scripts/estimation/export_estimator_policy.py``.

Inputs per step: the measured joint positions (server order).  The 320-D
history (joint_pos_rel + previous commands, 10 frames each) is maintained
exactly as in the parent class, so no other sensor is needed.  Outputs: the
next joint-position targets (as before) and ``self.last_estimate``, the
estimated cube state in the palm frame::

    {"pos": (3,), "quat_wxyz": (4,), "lin_vel": (3,), "ang_vel": (3,),
     "contact_prob": (4,)  # if, mf, rf, th}

Safety: this class only computes targets.  It never opens a serial port.
Physical runs must go through ``hardware/deploy_direct.py`` with the
supervised procedure in ``TA_GUIDE.md`` (shadow pass first, current and
temperature limits unchanged).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from policy.yaw_policy import ContinuousYawPolicy


class EstimatorYawPolicy(ContinuousYawPolicy):
  def __init__(self, bundle_path: str | Path, device: str = "cpu"):
    self._device = device
    self.last_estimate: dict[str, np.ndarray] | None = None
    super().__init__(bundle_path)

  def _load_weights(self, path: Path) -> None:
    import torch

    extra = {"meta.json": ""}
    self._bundle = torch.jit.load(str(path), map_location=self._device, _extra_files=extra)
    self._bundle.eval()
    self.meta = json.loads(extra["meta.json"] or "{}")
    if self.meta.get("format") != "leap_estimator_policy/v1":
      raise ValueError(f"{path} is not a leap_estimator_policy/v1 bundle")
    if self.meta.get("privileged_inputs", True):
      raise ValueError("Bundle declares privileged inputs; refusing to deploy.")

  def _forward(self, obs_320: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
      x = torch.as_tensor(obs_320, dtype=torch.float32, device=self._device).reshape(1, 320)
      action, state = self._bundle(x)
    s = state[0].cpu().numpy()
    self.last_estimate = {
      "pos": s[0:3],
      "quat_wxyz": s[3:7],
      "lin_vel": s[7:10],
      "ang_vel": s[10:13],
      "contact_prob": s[13:17],
    }
    return action[0].cpu().numpy().reshape(-1)

  def estimate_from_history(self) -> dict[str, np.ndarray] | None:
    """Estimated cube state for the current history (no action applied)."""
    self._forward(self._get_chronological_obs())
    return self.last_estimate
