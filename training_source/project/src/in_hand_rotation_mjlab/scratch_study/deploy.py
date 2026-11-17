"""Export from-scratch study policies (A / C) to robot-side TorchScript bundles.

Why a separate exporter: the study policies differ from ``model_400`` and the
earlier ``estimation/deploy.py`` bundles in three ways that matter on hardware.

1. **Action squash.**  The study environment executes ``tanh(u)``
   (``leap_right_scratch_study/actions.py``); ``model_400`` executes
   ``clip(u, -1, 1)``.  Policy std grew to ~1.15 during training, so the raw
   outputs are large and the two mappings differ substantially.
2. **Observation normalisation.**  RSL-RL 4.0.1 ``EmpiricalNormalization``
   uses ``(x - mean) / (std + 1e-2)``.  The legacy NumPy runner uses
   ``std + 1e-8`` (``docs/state_estimation.md`` Section 2.2).  The bundle carries
   the training normalisation itself.
3. **C features.**  Condition C appends ``features.cube_state_features`` of the
   frozen estimator B; the estimator is *not* stored in the policy checkpoint
   (``actors.py``), so it must be attached here, from the same file used in
   training (sha256 recorded).

Bundle (``leap_scratch_policy/v1``), one input ``obs_320`` [N, 320]::

    raw_action   [N, 16]  actor mean u (pre-squash)
    action       [N, 16]  executed normalised action tanh(u) in [-1, 1]
    cube_state   [N, 17]  estimator B: pos(3, m) | quat wxyz(4) | lin_vel(3) |
                          ang_vel(3) | contact prob(4: if, mf, rf, th); palm frame
    features     [N, 19]  the 19 cube-state features (what C's actor sees)
    forces       [N, 4]   fingertip normal force [N] (thumb, index, middle, ring)
                          from the force estimator, or zeros if none attached

Cube state, features and forces are produced for *both* A and C, so the two
policies are logged with identical monitors on hardware.  For A they are
monitors only and never reach the actor.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn

from in_hand_rotation_mjlab.estimation.layout import ACTOR_OBS_DIM, FINGERTIP_NAMES, JOINT_NAMES
from in_hand_rotation_mjlab.estimation.model import CubeStateEstimator, load_estimator
from in_hand_rotation_mjlab.estimation.rotations import matrix_to_quat

from .features import FEATURE_DIM, FEATURE_NAMES, cube_state_features

BUNDLE_FORMAT = "leap_scratch_policy/v1"
NORMALIZER_EPS = 1e-2  # rsl_rl.modules.EmpiricalNormalization default (4.0.1)
DELTA_RANGE_RAD = 1.0 / 24.0
STATE_LAYOUT = ["pos_x", "pos_y", "pos_z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz"] + [
  f"contact_{f}" for f in FINGERTIP_NAMES
]
FORCE_LAYOUT = ["thumb_N", "index_N", "middle_N", "ring_N"]


def sha256_file(path: str | Path) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def _mlp_from_state(actor_state: dict[str, torch.Tensor]) -> nn.Sequential:
  """Rebuild RSL-RL's ``MLP`` (Linear/ELU alternating, indices 0,2,4,6)."""
  idx = sorted({int(k.split(".")[1]) for k in actor_state if k.startswith("mlp.") and k.endswith(".weight")})
  if idx != [0, 2, 4, 6]:
    raise ValueError(f"unexpected actor layout, linear layers at {idx}")
  layers: list[nn.Module] = []
  for j, i in enumerate(idx):
    w = actor_state[f"mlp.{i}.weight"]
    lin = nn.Linear(w.shape[1], w.shape[0])
    lin.weight.data.copy_(w)
    lin.bias.data.copy_(actor_state[f"mlp.{i}.bias"])
    layers.append(lin)
    if j < len(idx) - 1:
      layers.append(nn.ELU())
  return nn.Sequential(*layers).eval()


class _ZeroForces(nn.Module):
  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return torch.zeros(obs.shape[0], 4, dtype=obs.dtype, device=obs.device)


class ScratchDeployedPolicy(nn.Module):
  def __init__(
    self,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    mlp: nn.Module,
    estimator: CubeStateEstimator,
    force_model: nn.Module | None,
    use_features: bool,
  ) -> None:
    super().__init__()
    self.register_buffer("obs_mean", obs_mean.reshape(1, ACTOR_OBS_DIM).clone())
    self.register_buffer("obs_std", obs_std.reshape(1, ACTOR_OBS_DIM).clone())
    self.mlp = mlp
    self.estimator = estimator.eval()
    self.force_model = (force_model or _ZeroForces()).eval()
    self.use_features = use_features

  def forward(self, obs: torch.Tensor):
    z = (obs - self.obs_mean) / (self.obs_std + NORMALIZER_EPS)
    e = self.estimator(obs)
    f = cube_state_features(e["pos"], e["rotmat"], e["lin_vel"], e["ang_vel"], e["contact_prob"])
    if self.use_features:
      z = torch.cat([z, f], dim=-1)
    u = self.mlp(z)
    state = torch.cat([e["pos"], matrix_to_quat(e["rotmat"]), e["lin_vel"], e["ang_vel"], e["contact_prob"]], -1)
    return u, torch.tanh(u), state, f, self.force_model(obs)


def build_policy(
  policy_checkpoint: str | Path,
  estimator_checkpoint: str | Path,
  force_checkpoint: str | Path | None = None,
) -> tuple[ScratchDeployedPolicy, str]:
  blob = torch.load(policy_checkpoint, map_location="cpu", weights_only=False)
  sd = blob["actor_state_dict"]
  in_dim = sd["mlp.0.weight"].shape[1]
  if in_dim == ACTOR_OBS_DIM:
    condition = "A"
  elif in_dim == ACTOR_OBS_DIM + FEATURE_DIM:
    condition = "C"
  else:
    raise ValueError(f"actor input {in_dim}: not an A (320) or C (339) study actor")
  if any(k.startswith("estimator.") for k in sd):
    raise ValueError("checkpoint carries an embedded estimator; this is not a scratch-study actor")
  est = load_estimator(estimator_checkpoint).eval()
  force = None
  if force_checkpoint is not None:
    from in_hand_rotation_mjlab.force_estimation.model import load_force_estimator

    force = load_force_estimator(force_checkpoint).eval()
  for m in (est, force):
    if m is not None:
      for p in m.parameters():
        p.requires_grad_(False)
  model = ScratchDeployedPolicy(
    sd["obs_normalizer._mean"], sd["obs_normalizer._std"], _mlp_from_state(sd), est, force, condition == "C"
  ).eval()
  return model, condition


def export_bundle(
  policy_checkpoint: str | Path,
  output: str | Path,
  estimator_checkpoint: str | Path,
  force_checkpoint: str | Path | None = None,
  label: str | None = None,
) -> dict:
  model, condition = build_policy(policy_checkpoint, estimator_checkpoint, force_checkpoint)
  g = torch.Generator().manual_seed(0)
  # Trace on realistic-scale inputs (the cube canonicalisation uses argmax, which
  # traces as a tensor op; there is no Python control flow on values).
  example = model.obs_mean + 0.05 * torch.randn(4, ACTOR_OBS_DIM, generator=g)
  with torch.no_grad():
    traced = torch.jit.trace(model, example, check_trace=False)
    probe = model.obs_mean + 0.2 * torch.randn(64, ACTOR_OBS_DIM, generator=g)
    for a, b in zip(model(probe), traced(probe)):
      if not torch.allclose(a, b, atol=1e-5):
        raise RuntimeError("traced bundle disagrees with eager model")
  meta = {
    "format": BUNDLE_FORMAT,
    "condition": condition,
    "label": label or Path(policy_checkpoint).stem,
    "source_checkpoint": Path(policy_checkpoint).name,
    "source_checkpoint_sha256": sha256_file(policy_checkpoint),
    "estimator": Path(estimator_checkpoint).name,
    "estimator_sha256": sha256_file(estimator_checkpoint),
    "estimator_role": "actor input (C)" if condition == "C" else "monitor only (A)",
    "force_estimator": Path(force_checkpoint).name if force_checkpoint else None,
    "force_estimator_sha256": sha256_file(force_checkpoint) if force_checkpoint else None,
    "obs_layout": "320 = joint_pos_rel[10x16 oldest->newest] | prev_command_abs[10x16 oldest->newest]",
    "joint_order": list(JOINT_NAMES),
    "outputs": ["raw_action", "action", "cube_state", "features", "forces"],
    "state_layout": STATE_LAYOUT,
    "feature_layout": list(FEATURE_NAMES),
    "force_layout": FORCE_LAYOUT,
    "normalizer_eps": NORMALIZER_EPS,
    "action_squash": "tanh",
    "action": "tanh(raw) * 1/24 rad added to previous command, then joint limits",
    "delta_range_rad": DELTA_RANGE_RAD,
    "privileged_inputs": False,
  }
  output = Path(output)
  output.parent.mkdir(parents=True, exist_ok=True)
  traced.save(str(output), _extra_files={"meta.json": json.dumps(meta)})
  loaded = torch.jit.load(str(output))
  with torch.no_grad():
    for a, b in zip(model(probe), loaded(probe)):
      if not torch.allclose(a, b, atol=1e-5):
        raise RuntimeError("saved bundle disagrees with eager model")
  return meta
