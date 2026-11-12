"""Unit tests for the estimation extension that need only PyTorch + RSL-RL.

Run:  python -m pytest tests/test_estimation.py -q
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from in_hand_rotation_mjlab.estimation import layout
from in_hand_rotation_mjlab.estimation.losses import (
  EstimatorLoss,
  LossWeights,
  MetricAccumulator,
)
from in_hand_rotation_mjlab.estimation.migration import (
  ACTOR_CLASS,
  infer_actor_cfg,
  migrate_checkpoint,
)
from in_hand_rotation_mjlab.estimation.model import (
  CubeStateEstimator,
  EstimatorConfig,
  load_estimator,
  save_estimator,
)
from in_hand_rotation_mjlab.estimation.rotations import (
  cube_symmetry_matrices,
  geodesic_angle,
  matrix_to_quat,
  matrix_to_rot6d,
  quat_to_matrix,
  rot6d_to_matrix,
  symmetric_chordal_loss,
  symmetric_geodesic_angle,
  tilt_angle,
)

REPO = Path(__file__).resolve().parents[3]
MODEL_400 = REPO / "checkpoint" / "model_400.pt"


def random_quat(n: int, gen=None) -> torch.Tensor:
  q = torch.randn(n, 4, generator=gen)
  return q / q.norm(dim=-1, keepdim=True)


def axis_angle_quat(axis, angle) -> torch.Tensor:
  axis = torch.tensor(axis, dtype=torch.float32)
  axis = axis / axis.norm()
  return torch.cat([torch.tensor([math.cos(angle / 2)]), axis * math.sin(angle / 2)])


# ----------------------------------------------------------------- layout
def test_observation_layout_constants():
  assert layout.ACTOR_OBS_DIM == 320
  assert layout.JOINT_POS_REL_SLICE == slice(0, 160)
  assert layout.PREV_COMMAND_SLICE == slice(160, 320)
  assert layout.FINGERTIP_NAMES == ("if", "mf", "rf", "th")
  assert layout.FEATURE_MODES == {"full": 19, "contact": 4}


def test_split_actor_obs_matches_hardware_runner():
  """The portable hardware runner builds the same 320-D layout."""
  import sys

  sys.path.insert(0, str(REPO))
  from policy.yaw_policy import ContinuousYawPolicy

  pol = ContinuousYawPolicy(REPO / "checkpoint" / "policy_weights.npz")
  pol.reset()
  rng = np.random.default_rng(0)
  qs = []
  for _ in range(12):
    q = pol.default_joint_pos + 0.05 * rng.standard_normal(16).astype(np.float32)
    qs.append(q)
    pol.step(q)
  obs = torch.from_numpy(pol._get_chronological_obs()[0])
  q_rel, cmd = layout.split_actor_obs(obs)
  assert q_rel.shape == cmd.shape == (10, 16)
  # newest joint frame is the last measured q relative to the default pose
  np.testing.assert_allclose(q_rel[-1].numpy(), qs[-1] - pol.default_joint_pos, atol=1e-6)
  np.testing.assert_allclose(q_rel[0].numpy(), qs[-10] - pol.default_joint_pos, atol=1e-6)


# -------------------------------------------------------------- rotations
def test_quat_matrix_roundtrip_and_6d():
  q = random_quat(256, torch.Generator().manual_seed(0))
  r = quat_to_matrix(q)
  assert torch.allclose(r @ r.transpose(-1, -2), torch.eye(3).expand(256, 3, 3), atol=1e-5)
  assert torch.allclose(torch.linalg.det(r), torch.ones(256), atol=1e-5)
  q2 = matrix_to_quat(r)
  # q and -q are the same rotation
  assert torch.allclose((q * q2).sum(-1).abs(), torch.ones(256), atol=1e-5)
  r6 = rot6d_to_matrix(matrix_to_rot6d(r))
  assert torch.allclose(r6, r, atol=1e-5)
  # Gram-Schmidt of an arbitrary 6-vector is a proper rotation
  g = rot6d_to_matrix(torch.randn(64, 6))
  assert torch.allclose(torch.linalg.det(g), torch.ones(64), atol=1e-5)


def test_quat_to_matrix_known_rotation():
  r = quat_to_matrix(axis_angle_quat([0, 0, 1], math.pi / 2)[None])[0]
  expected = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
  assert torch.allclose(r, expected, atol=1e-6)


def test_cube_symmetry_group():
  s = cube_symmetry_matrices()
  assert s.shape == (24, 3, 3)
  assert torch.allclose(torch.linalg.det(s), torch.ones(24))
  # closed under multiplication
  prods = (s[:, None] @ s[None]).reshape(-1, 9)
  flat = s.reshape(24, 9)
  assert all(((flat - p).abs().sum(-1) < 1e-6).any() for p in prods)


def test_symmetric_error_ignores_cube_symmetries():
  s = cube_symmetry_matrices()
  r = quat_to_matrix(random_quat(24, torch.Generator().manual_seed(1)))
  r_sym = r @ s  # each sample rotated by a different symmetry
  ang = symmetric_geodesic_angle(r, r_sym, s)
  assert ang.max() < 1e-3
  loss, _ = symmetric_chordal_loss(r, r_sym, s)
  assert loss.max() < 1e-5
  # ... but a 20 degree yaw error is still 20 degrees
  rz = quat_to_matrix(axis_angle_quat([0, 0, 1], math.radians(20))[None]).expand(24, 3, 3)
  ang = symmetric_geodesic_angle(r @ rz, r, s)
  assert torch.allclose(torch.rad2deg(ang), torch.full((24,), 20.0), atol=1e-2)
  raw = geodesic_angle(r, r_sym)
  assert raw.max() > 1.0  # raw error does see the 90/180 deg symmetries


def test_tilt_angle():
  assert float(tilt_angle(torch.eye(3)[None])) < 1e-3
  r = quat_to_matrix(axis_angle_quat([1, 0, 0], math.radians(30))[None])
  assert abs(math.degrees(float(tilt_angle(r))) - 30.0) < 1e-3
  r = quat_to_matrix(axis_angle_quat([1, 0, 0], math.radians(90))[None])
  assert float(tilt_angle(r)) < 1e-3  # another face is flat


# ------------------------------------------------------------------ model
def make_batch(n: int, gen=None) -> dict[str, torch.Tensor]:
  return {
    "obs": torch.randn(n, 320, generator=gen),
    "pos": 0.03 * torch.randn(n, 3, generator=gen),
    "quat": random_quat(n, gen),
    "lin_vel": 0.05 * torch.randn(n, 3, generator=gen),
    "ang_vel": 0.3 * torch.randn(n, 3, generator=gen),
    "contact": (torch.rand(n, 4, generator=gen) > 0.5).to(torch.uint8),
    "contact_valid": torch.ones(n, dtype=torch.uint8),
  }


def test_estimator_output_shapes_and_ranges():
  model = CubeStateEstimator(EstimatorConfig(hidden_dims=(64, 64)))
  out = model.predict(torch.randn(7, 320))
  assert out["pos"].shape == (7, 3)
  assert out["rotmat"].shape == (7, 3, 3)
  assert out["quat"].shape == (7, 4)
  assert torch.allclose(out["quat"].norm(dim=-1), torch.ones(7), atol=1e-5)
  assert out["lin_vel"].shape == out["ang_vel"].shape == (7, 3)
  assert out["contact_prob"].shape == (7, 4)
  assert ((out["contact_prob"] >= 0) & (out["contact_prob"] <= 1)).all()
  full = model.policy_features(torch.randn(7, 320), "full")
  assert full.shape == (7, 19)
  assert torch.allclose(full[:, 3:6].norm(dim=-1), torch.ones(7), atol=1e-5)
  assert model.policy_features(torch.randn(7, 320), "contact").shape == (7, 4)
  with pytest.raises(ValueError):
    model(torch.randn(3, 83))  # privileged critic-sized input is rejected


def test_estimator_input_is_actor_obs_only():
  """The estimator signature only admits the 320-D proprioceptive vector."""
  import inspect

  sig = inspect.signature(CubeStateEstimator.forward)
  assert list(sig.parameters) == ["self", "obs"]
  assert CubeStateEstimator().cfg.obs_dim == layout.ACTOR_OBS_DIM


def test_estimator_save_load_roundtrip(tmp_path):
  model = CubeStateEstimator(EstimatorConfig(hidden_dims=(32,)))
  model.set_normalization(
    torch.randn(320), torch.rand(320) + 0.1, torch.randn(3), torch.rand(3) + 0.1,
    torch.rand(3) + 0.1, torch.rand(3) + 0.1,
  )
  save_estimator(model, tmp_path / "e.pt")
  loaded = load_estimator(tmp_path / "e.pt")
  x = torch.randn(5, 320)
  assert torch.allclose(model(x)["pos"], loaded(x)["pos"])


def test_loss_decreases_and_contact_mask():
  gen = torch.Generator().manual_seed(0)
  batch = make_batch(512, gen)
  model = CubeStateEstimator(EstimatorConfig(hidden_dims=(128, 128)))
  loss_fn = EstimatorLoss(LossWeights(), "cpu")
  opt = torch.optim.Adam(model.parameters(), lr=3e-3)
  first = None
  for _ in range(150):
    loss, parts = loss_fn(model, model(batch["obs"]), batch)
    first = first if first is not None else float(loss.detach())
    opt.zero_grad()
    loss.backward()
    opt.step()
  assert float(loss) < 0.5 * first
  # invalid rows do not contribute to the contact loss
  b2 = dict(batch)
  b2["contact_valid"] = torch.zeros(512, dtype=torch.uint8)
  _, parts = loss_fn(model, model(batch["obs"]), b2)
  assert float(parts["loss/contact"]) == 0.0


def test_metrics_perfect_prediction():
  batch = make_batch(100, torch.Generator().manual_seed(2))
  r = quat_to_matrix(batch["quat"])
  pred = {
    "pos": batch["pos"].clone(),
    "rotmat": r @ cube_symmetry_matrices()[5],  # symmetric copy
    "lin_vel": batch["lin_vel"].clone(),
    "ang_vel": batch["ang_vel"].clone(),
    "contact_prob": batch["contact"].float(),
  }
  acc = MetricAccumulator("cpu")
  acc.update(pred, batch)
  m = acc.compute()
  assert m["pos_rmse_mm"] < 1e-4
  assert m["rot_err_sym_mean_deg"] < 0.1
  assert m["contact_micro_f1"] == pytest.approx(1.0)
  assert m["contact_micro_precision"] == pytest.approx(1.0)


# --------------------------------------------- policy / checkpoint migration
def _obs_td(n: int = 4) -> TensorDict:
  return TensorDict(
    {"actor": torch.randn(n, 320), "critic": torch.randn(n, 83)}, batch_size=[n]
  )


def _actor_kwargs():
  return dict(
    hidden_dims=(512, 512, 256), activation="elu", obs_normalization=True,
    init_noise_std=0.7, noise_std_type="scalar", stochastic=True,
  )


@pytest.mark.skipif(not MODEL_400.is_file(), reason="model_400.pt missing")
@pytest.mark.parametrize("mode,fdim", [("full", 19), ("contact", 4)])
def test_migration_preserves_model_400_actions(tmp_path, mode, fdim):
  from rsl_rl.models.mlp_model import MLPModel

  from in_hand_rotation_mjlab.estimation.policy import EstimatorAugmentedMLPModel

  est = CubeStateEstimator()
  est.set_normalization(
    torch.zeros(320), torch.ones(320), torch.zeros(3), torch.full((3,), 0.02),
    torch.full((3,), 0.05), torch.full((3,), 0.3),
  )
  save_estimator(est, tmp_path / "est.pt")
  out = migrate_checkpoint(MODEL_400, tmp_path / "est.pt", mode, tmp_path / "aug.pt")
  src = torch.load(MODEL_400, map_location="cpu", weights_only=False)

  groups = {"actor": ["actor"], "critic": ["critic"]}
  torch.manual_seed(0)
  obs = _obs_td(16)
  base = MLPModel(obs, groups, "actor", 16, **_actor_kwargs())
  base.load_state_dict(src["actor_state_dict"])
  aug = EstimatorAugmentedMLPModel(
    obs, groups, "actor", 16, estimator_config=out["estimation"]["actor_cfg"]["estimator_config"],
    feature_mode=mode, **_actor_kwargs(),
  )
  aug.load_state_dict(out["actor_state_dict"], strict=True)
  base.eval(), aug.eval()
  assert aug.mlp[0].weight.shape == (512, 320 + fdim)
  with torch.no_grad():
    a, b = base(obs), aug(obs)
    # Raw outputs reach |a| ~ 30 (the env clips to +-1), so float32 summation
    # order alone gives ~1e-4 differences; compare relatively, and require the
    # executed (clipped) actions to match tightly.
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-4), (a - b).abs().max()
    assert torch.allclose(a.clamp(-1, 1), b.clamp(-1, 1), atol=1e-4)
  # estimator frozen, policy MLP trainable
  assert not any(p.requires_grad for p in aug.estimator.parameters())
  assert all(p.requires_grad for p in aug.mlp.parameters())
  # cfg inference from the bare state dict (as saved by RSL-RL during PPO)
  inferred = infer_actor_cfg(out["actor_state_dict"])
  assert inferred["feature_mode"] == mode
  assert inferred["class_name"] == ACTOR_CLASS
  assert tuple(inferred["estimator_config"]["hidden_dims"]) == (512, 512, 256)
  assert infer_actor_cfg(src["actor_state_dict"]) is None


@pytest.mark.skipif(not MODEL_400.is_file(), reason="model_400.pt missing")
def test_migrated_optimizer_matches_ppo_param_order(tmp_path):
  from itertools import chain

  from rsl_rl.models.mlp_model import MLPModel

  from in_hand_rotation_mjlab.estimation.policy import EstimatorAugmentedMLPModel

  save_estimator(CubeStateEstimator(), tmp_path / "est.pt")
  out = migrate_checkpoint(MODEL_400, tmp_path / "est.pt", "full", tmp_path / "aug.pt")
  groups = {"actor": ["actor"], "critic": ["critic"]}
  obs = _obs_td()
  aug = EstimatorAugmentedMLPModel(
    obs, groups, "actor", 16, estimator_config=out["estimation"]["actor_cfg"]["estimator_config"],
    feature_mode="full", **_actor_kwargs(),
  )
  critic = MLPModel(obs, groups, "critic", 1, hidden_dims=(512, 512, 256), obs_normalization=True)
  aug.load_state_dict(out["actor_state_dict"])
  critic.load_state_dict(out["critic_state_dict"])
  # Same construction as rsl_rl.algorithms.PPO.__init__
  opt = torch.optim.Adam(chain(aug.parameters(), critic.parameters()), lr=3e-4)
  opt.load_state_dict(out["optimizer_state_dict"])
  params = list(chain(aug.parameters(), critic.parameters()))
  for i, p in enumerate(params):
    st = opt.state.get(p)
    if st:
      assert st["exp_avg"].shape == p.shape, i
  # first actor layer moments padded, estimator parameters have no state
  w0 = aug.mlp[0].weight
  assert opt.state[w0]["exp_avg"][:, 320:].abs().sum() == 0
  assert all(p not in opt.state for p in aug.estimator.parameters())
  # one optimisation step runs (frozen params are skipped)
  loss = aug(obs).sum() + critic(obs).sum()
  loss.backward()
  opt.step()


# ------------------------------------------------------------- deployment
@pytest.mark.skipif(not MODEL_400.is_file(), reason="model_400.pt missing")
def test_export_bundle_and_robot_runner(tmp_path):
  import sys

  from in_hand_rotation_mjlab.estimation.deploy import export_bundle

  sys.path.insert(0, str(REPO))
  from policy.estimator_policy import EstimatorYawPolicy
  from policy.yaw_policy import ContinuousYawPolicy

  est = CubeStateEstimator()
  save_estimator(est, tmp_path / "est.pt")
  # (a) model_400 + estimator as a monitor: actions must equal the NumPy runner
  meta = export_bundle(MODEL_400, tmp_path / "monitor.pt", tmp_path / "est.pt")
  assert meta["privileged_inputs"] is False
  base = ContinuousYawPolicy(REPO / "checkpoint" / "policy_weights.npz")
  # RSL-RL's EmpiricalNormalization divides by (std + 1e-2); the NumPy runner
  # in policy/yaw_policy.py uses 1e-8.  With eps matched, outputs agree.
  base._eps = 1e-2
  mon = EstimatorYawPolicy(tmp_path / "monitor.pt")
  q0 = base.reset()
  mon.reset(q0)
  rng = np.random.default_rng(0)
  q = q0.copy()
  for _ in range(15):
    t_base, raw_base, _ = base.step(q)
    t_mon, raw_mon, _ = mon.step(q)
    np.testing.assert_allclose(raw_mon, raw_base, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(t_mon, t_base, atol=1e-5)
    q = t_base + 0.01 * rng.standard_normal(16).astype(np.float32)
  e = mon.last_estimate
  assert e["pos"].shape == (3,) and e["quat_wxyz"].shape == (4,) and e["contact_prob"].shape == (4,)
  assert abs(np.linalg.norm(e["quat_wxyz"]) - 1) < 1e-4
  # (b) augmented checkpoint exports and runs with the same interface
  migrate_checkpoint(MODEL_400, tmp_path / "est.pt", "full", tmp_path / "aug.pt")
  meta = export_bundle(tmp_path / "aug.pt", tmp_path / "aug_bundle.pt")
  assert meta["feature_mode"] == "full"
  aug = EstimatorYawPolicy(tmp_path / "aug_bundle.pt")
  aug.reset(q0)
  target, raw, delta = aug.step(q0)
  assert target.shape == (16,) and np.max(np.abs(delta)) <= aug.delta_range + 1e-7
