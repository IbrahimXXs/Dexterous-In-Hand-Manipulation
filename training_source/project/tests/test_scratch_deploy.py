"""Exported scratch-study bundles reproduce the training-time actors exactly.

Reference = the RSL-RL actor classes the study trained with
(``MLPModel`` for A, ``scratch_study.actors.EstimatedStateMLPModel`` for C),
loaded from the selected checkpoints, followed by the study's tanh squash.
CPU only; no MJLab needed (skips if rsl_rl is not installed).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

rsl = pytest.importorskip("rsl_rl")
from tensordict import TensorDict  # noqa: E402

from in_hand_rotation_mjlab.estimation.layout import ACTOR_OBS_DIM  # noqa: E402
from in_hand_rotation_mjlab.scratch_study.deploy import build_policy, export_bundle  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
STUDY = PROJECT / "experiments" / "scratch_state_estimation"
EST = STUDY / "models" / "estimator_B.pt"
FORCE = PROJECT / "experiments" / "force_estimation" / "models" / "force_estimator.pt"
CKPTS = sorted((STUDY / "checkpoints").glob("[AC]_seed*_model_*.pt"))


def _reference_actor(ckpt: Path):
  from rsl_rl.models.mlp_model import MLPModel

  from in_hand_rotation_mjlab.scratch_study.actors import EstimatedStateMLPModel

  sd = torch.load(ckpt, map_location="cpu", weights_only=False)["actor_state_dict"]
  obs = TensorDict({"actor": torch.zeros(1, ACTOR_OBS_DIM)}, batch_size=[1])
  kw = dict(hidden_dims=[512, 512, 256], activation="elu", obs_normalization=True, stochastic=True,
            init_noise_std=0.7, noise_std_type="scalar")
  if ckpt.name.startswith("C"):
    actor = EstimatedStateMLPModel(obs, {"actor": ["actor"]}, "actor", 16, estimator_path=str(EST), **kw)
  else:
    actor = MLPModel(obs, {"actor": ["actor"]}, "actor", 16, **kw)
  actor.load_state_dict(sd, strict=True)
  return actor.eval()


def _probe_obs(n: int = 512) -> torch.Tensor:
  sd = torch.load(CKPTS[0], map_location="cpu", weights_only=False)["actor_state_dict"]
  g = torch.Generator().manual_seed(1)
  return sd["obs_normalizer._mean"] + sd["obs_normalizer._std"] * torch.randn(n, ACTOR_OBS_DIM, generator=g)


@pytest.mark.parametrize("ckpt", CKPTS, ids=lambda p: p.stem)
def test_bundle_matches_training_actor(ckpt: Path, tmp_path: Path):
  obs = _probe_obs()
  ref = _reference_actor(ckpt)
  with torch.no_grad():
    u_ref = ref(TensorDict({"actor": obs}, batch_size=[obs.shape[0]]))
  meta = export_bundle(ckpt, tmp_path / "b.pt", EST, FORCE)
  bundle = torch.jit.load(str(tmp_path / "b.pt"))
  with torch.no_grad():
    u, a, state, feats, forces = bundle(obs)
  assert meta["condition"] == ckpt.name[0]
  assert torch.allclose(u, u_ref, atol=1e-4), (u - u_ref).abs().max()
  assert torch.allclose(a, torch.tanh(u_ref), atol=1e-5)
  if ckpt.name.startswith("C"):
    assert torch.allclose(feats, ref.last_features, atol=1e-5)
  assert state.shape == (obs.shape[0], 17) and forces.shape == (obs.shape[0], 4)
  assert torch.all(forces >= 0)


def test_estimator_features_drive_C_but_not_A():
  obs = _probe_obs(64)
  a_model, _ = build_policy(next(c for c in CKPTS if c.name.startswith("A")), EST)
  c_model, _ = build_policy(next(c for c in CKPTS if c.name.startswith("C")), EST)
  with torch.no_grad():
    for m, should_change in ((a_model, False), (c_model, True)):
      u0 = m(obs)[0]
      w = m.estimator.head.bias.data.clone()
      m.estimator.head.bias.data += 0.5
      u1 = m(obs)[0]
      m.estimator.head.bias.data.copy_(w)
      assert bool((u1 - u0).abs().max() > 1e-3) == should_change
