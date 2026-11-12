"""Export the estimator(-augmented) policy for real-robot inference.

The exported TorchScript bundle has a single input, the 320-D actor
observation built from quantities the real LEAP Hand provides (measured joint
positions and the policy's own previous commands, see ``layout.py``), and
returns::

    actions  [N, 16]   raw policy output (clip to [-1, 1], scale by 1/24 rad)
    state    [N, 17]   pos(3, m) | quat wxyz(4) | lin_vel(3) | ang_vel(3) |
                       contact probability(4: if, mf, rf, th)   palm frame

No simulator, privileged or camera quantity is an input.  The bundle needs
only ``torch`` (no MJLab / RSL-RL), so ``policy/estimator_policy.py`` can run
it next to the existing hardware bridge.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from .layout import ACTOR_OBS_DIM, FINGERTIP_NAMES, JOINT_NAMES
from .model import CubeStateEstimator, EstimatorConfig, load_estimator
from .rotations import matrix_to_quat

STATE_LAYOUT = ["pos_x", "pos_y", "pos_z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz"] + [
  f"contact_{f}" for f in FINGERTIP_NAMES
]


class _StateHead(nn.Module):
  def __init__(self, est: CubeStateEstimator):
    super().__init__()
    self.est = est

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    o = self.est(obs)
    return torch.cat(
      [o["pos"], matrix_to_quat(o["rotmat"]), o["lin_vel"], o["ang_vel"], o["contact_prob"]], -1
    )


class DeployedPolicy(nn.Module):
  """obs_320 -> (actions, state).  Actor may be the 320-input baseline."""

  def __init__(self, normalizer: nn.Module, mlp: nn.Module, est: CubeStateEstimator, feature_mode: str | None):
    super().__init__()
    self.normalizer = normalizer
    self.mlp = mlp
    self.state = _StateHead(est)
    self.est = est
    self.feature_mode = feature_mode or "none"

  def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = self.normalizer(obs)
    if self.feature_mode != "none":
      z = torch.cat([z, self.est.policy_features(obs, self.feature_mode)], -1)
    return self.mlp(z), self.state(obs)


def _actor_parts(actor_state: dict) -> tuple[nn.Module, nn.Module]:
  from rsl_rl.modules import MLP, EmpiricalNormalization

  w = [actor_state[f"mlp.{i}.weight"] for i in (0, 2, 4, 6)]
  norm = EmpiricalNormalization(ACTOR_OBS_DIM)
  norm.load_state_dict({k.split(".", 1)[1]: v for k, v in actor_state.items() if k.startswith("obs_normalizer.")})
  mlp = MLP(w[0].shape[1], w[3].shape[0], [w[0].shape[0], w[1].shape[0], w[2].shape[0]], "elu")
  mlp.load_state_dict({k.split(".", 1)[1]: v for k, v in actor_state.items() if k.startswith("mlp.")})
  return norm.eval(), mlp.eval()


def export_bundle(
  policy_checkpoint: str | Path,
  output: str | Path,
  estimator_checkpoint: str | Path | None = None,
) -> dict:
  """Export ``policy_checkpoint`` (augmented, or model_400 + a separate
  estimator used only as a state monitor) to a TorchScript bundle."""
  from .migration import infer_actor_cfg

  blob = torch.load(policy_checkpoint, map_location="cpu", weights_only=False)
  actor_state = blob["actor_state_dict"]
  cfg = infer_actor_cfg(actor_state)
  if cfg is not None:
    ecfg = dict(cfg["estimator_config"])
    ecfg["hidden_dims"] = tuple(ecfg["hidden_dims"])
    est = CubeStateEstimator(EstimatorConfig(**ecfg))
    est.load_state_dict({k[len("estimator."):]: v for k, v in actor_state.items() if k.startswith("estimator.")})
    feature_mode = cfg["feature_mode"]
    actor_state = {k: v for k, v in actor_state.items() if not k.startswith("estimator.")}
  else:
    if estimator_checkpoint is None:
      raise ValueError("Baseline actor: pass --estimator to attach a state monitor")
    est = load_estimator(estimator_checkpoint)
    feature_mode = None
  est.eval()
  norm, mlp = _actor_parts(actor_state)
  model = DeployedPolicy(norm, mlp, est, feature_mode).eval()
  example = torch.zeros(1, ACTOR_OBS_DIM)
  with torch.no_grad():
    ref_a, ref_s = model(example)
    traced = torch.jit.trace(model, example, check_trace=False)
    a2, s2 = traced(torch.randn(3, ACTOR_OBS_DIM))
    a3, s3 = model(torch.zeros(3, ACTOR_OBS_DIM))
  meta = {
    "format": "leap_estimator_policy/v1",
    "source_checkpoint": str(Path(policy_checkpoint).resolve()),
    "feature_mode": feature_mode or "none (baseline actor, estimator is a monitor only)",
    "obs_layout": "320 = joint_pos_rel[10x16 oldest->newest] | prev_command_abs[10x16]",
    "joint_order": list(JOINT_NAMES),
    "state_layout": STATE_LAYOUT,
    "action": "clip(raw,-1,1) * 1/24 rad added to previous command, then joint limits",
    "privileged_inputs": False,
  }
  output = Path(output)
  output.parent.mkdir(parents=True, exist_ok=True)
  traced.save(str(output), _extra_files={"meta.json": json.dumps(meta)})
  # round-trip check
  loaded = torch.jit.load(str(output))
  with torch.no_grad():
    la, ls = loaded(torch.zeros(1, ACTOR_OBS_DIM))
  assert torch.allclose(la, ref_a, atol=1e-5) and torch.allclose(ls, ref_s, atol=1e-5)
  return meta
