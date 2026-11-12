"""CPU tests for the from-scratch study (geometry, corrected terms, actors).

Run:  python -m pytest -q tests/test_scratch_study.py
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from in_hand_rotation_mjlab.scratch_study.geometry import (
  physical_tilt,
  quat_mul,
  rotation_vector,
  yaw_increment,
)

torch.manual_seed(0)


def axis_angle_quat(axis, angle) -> torch.Tensor:
  axis = torch.as_tensor(axis, dtype=torch.float64)
  axis = axis / axis.norm()
  angle = torch.as_tensor(angle, dtype=torch.float64)
  return torch.cat([torch.cos(angle / 2).reshape(-1, 1), axis * torch.sin(angle / 2).reshape(-1, 1)], -1)


def random_quats(n: int) -> torch.Tensor:
  q = torch.randn(n, 4, dtype=torch.float64)
  return q / q.norm(dim=-1, keepdim=True)


def euler_xyz_quat(roll, pitch, yaw) -> torch.Tensor:
  qx = axis_angle_quat([1, 0, 0], roll)
  qy = axis_angle_quat([0, 1, 0], pitch)
  qz = axis_angle_quat([0, 0, 1], yaw)
  return quat_mul(qz, quat_mul(qy, qx))


# ----------------------------------------------------------------- geometry
def test_physical_tilt_invariant_to_world_yaw_including_gimbal_lock():
  q0 = torch.cat([random_quats(200), euler_xyz_quat(0.3, math.radians(89.5), 1.0), euler_xyz_quat(2.0, -math.radians(89.9), -2.0)])
  for psi in torch.linspace(-math.pi, math.pi, 13):
    q = quat_mul(axis_angle_quat([0, 0, 1], psi).expand_as(q0), q0)
    assert physical_tilt(q, q0).abs().max() < 1e-6


@pytest.mark.parametrize("theta_deg", [1.0, 10.0, 30.0, 45.8, 80.0, 170.0])
def test_physical_tilt_equals_horizontal_rotation_angle(theta_deg):
  q0 = torch.cat([random_quats(100), euler_xyz_quat(0.0, math.radians(89.0), 0.0)])
  for phi in torch.linspace(0, 2 * math.pi, 7)[:-1]:
    axis = [math.cos(phi), math.sin(phi), 0.0]
    q = quat_mul(axis_angle_quat(axis, math.radians(theta_deg)).expand_as(q0), q0)
    assert torch.allclose(torch.rad2deg(physical_tilt(q, q0)), torch.full((q0.shape[0],), theta_deg, dtype=torch.float64), atol=1e-4)


def test_yaw_increment_exact_for_z_rotation_and_sign_invariant():
  q0 = random_quats(300)
  for d in (-3.0, -0.4, -1e-4, 0.0, 1e-4, 0.25, 3.0):
    q1 = quat_mul(axis_angle_quat([0, 0, 1], d).expand_as(q0), q0)
    assert torch.allclose(yaw_increment(q1, q0), torch.full((300,), d, dtype=torch.float64), atol=1e-7)
    assert torch.allclose(yaw_increment(-q1, q0), torch.full((300,), d, dtype=torch.float64), atol=1e-7)
    assert torch.allclose(yaw_increment(q1, -q0), torch.full((300,), d, dtype=torch.float64), atol=1e-7)


def test_yaw_increment_ignores_horizontal_rotation():
  q0 = random_quats(100)
  q1 = quat_mul(axis_angle_quat([1, 0, 0], 0.3).expand_as(q0), q0)
  assert yaw_increment(q1, q0).abs().max() < 1e-7


def test_rotation_vector_small_angle_limit():
  q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
  assert rotation_vector(q).abs().max() == 0


# ------------------------------------------------------- corrected MDP terms
class FakeCube:
  def __init__(self, n):
    self.data = SimpleNamespace(
      root_link_pos_w=torch.zeros(n, 3), root_link_quat_w=torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
      root_link_ang_vel_w=torch.zeros(n, 3),
    )


def fake_env(n=4):
  cube = FakeCube(n)
  return SimpleNamespace(scene={"cube": cube}, num_envs=n, device="cpu", step_dt=0.05,
                         episode_length_buf=torch.zeros(n, dtype=torch.long)), cube


def test_termination_anchor_is_fresh_and_tilt_is_physical():
  from in_hand_rotation_mjlab.scratch_study.mdp import cube_pose_deviation_physical

  env, cube = fake_env(3)
  term = cube_pose_deviation_physical(SimpleNamespace(params={}), env)
  # previous episode ended somewhere far away and tilted; reset marks only
  cube.data.root_link_pos_w[:] = torch.tensor([0.5, 0.0, 0.0])
  term.reset(None)
  # new episode: cube spawned near Euler gimbal lock (pitch 89 deg)
  q_start = euler_xyz_quat(0.4, math.radians(89.0), 0.0).float()
  cube.data.root_link_pos_w[:] = 0.0
  cube.data.root_link_quat_w[:] = q_start
  assert not term(env, 0.08, 0.8).any()  # anchoring step never terminates
  # rotate a full turn about world z in 10-degree steps: must never terminate
  for k in range(36):
    qz = axis_angle_quat([0, 0, 1], math.radians(10 * (k + 1))).float()
    cube.data.root_link_quat_w[:] = quat_mul(qz, q_start)
    assert not term(env, 0.08, 0.8).any()
  # 30 deg of physical tilt from the anchor: still fine; 50 deg: terminate
  for deg, expected in ((30.0, False), (50.0, True)):
    qt = axis_angle_quat([1, 0, 0], math.radians(deg)).float()
    cube.data.root_link_quat_w[:] = quat_mul(qt, q_start)
    assert bool(term(env, 0.08, 0.8).all()) is expected
  # position drift > 8 cm terminates
  cube.data.root_link_quat_w[:] = q_start
  cube.data.root_link_pos_w[:] = torch.tensor([0.09, 0.0, 0.0])
  assert term(env, 0.08, 0.8).all()


def test_reward_counts_genuine_rotation_and_never_reset_jumps():
  from in_hand_rotation_mjlab.scratch_study.mdp import genuine_yaw_rate_reward

  env, cube = fake_env(2)
  params = dict(object_name="cube", clip_min=-0.25, clip_max=0.25, history_steps=4)
  rew = genuine_yaw_rate_reward(SimpleNamespace(params=params), env)
  q = euler_xyz_quat(0.0, math.radians(88.0), 0.3).float()  # near gimbal lock
  cube.data.root_link_quat_w[:] = q
  rew(env, **params)  # anchoring step
  rates = []
  for k in range(8):  # +0.2 rad/s about world z  -> 0.01 rad per 0.05 s step
    q = quat_mul(axis_angle_quat([0, 0, 1], 0.01).float(), q)
    cube.data.root_link_quat_w[:] = q
    rates.append(rew(env, **params))
  assert torch.allclose(rates[-1], torch.full((2,), 0.2 * 1.0), atol=1e-4)
  # reset with a 120-degree yaw teleport: no reward from the jump
  rew.reset(torch.tensor([0, 1]))
  cube.data.root_link_quat_w[:] = quat_mul(axis_angle_quat([0, 0, 1], 2.1).float(), q)
  assert rew(env, **params).abs().max() < 1e-6


def test_drift_gate_uses_physical_tilt():
  from in_hand_rotation_mjlab.scratch_study.mdp import genuine_yaw_rate_reward

  env, cube = fake_env(1)
  params = dict(object_name="cube", clip_min=-0.25, clip_max=0.25, history_steps=1)
  rew = genuine_yaw_rate_reward(SimpleNamespace(params=params), env)
  q = euler_xyz_quat(0.0, math.radians(89.0), 0.0).float()
  cube.data.root_link_quat_w[:] = q
  rew(env, **params)
  # small physical wobble (5 deg) + rotation: full reward (Euler would see a large "tilt")
  q2 = quat_mul(axis_angle_quat([0, 0, 1], 0.005).float(), quat_mul(axis_angle_quat([1, 0, 0], math.radians(5)).float(), q))
  cube.data.root_link_quat_w[:] = q2
  r = rew(env, **params)
  assert r.item() == pytest.approx(0.1, abs=0.02)  # 0.005 rad / 0.05 s = 0.1 rad/s, gate inside


# --------------------------------------------------------- PPO / tanh action
def test_tanh_squash_keeps_ppo_ratio_exact():
  """log pi_tanh(a) = log N(u) - sum log(1 - tanh(u)^2): the Jacobian term does
  not depend on the policy, so the stored-Gaussian ratio is the exact ratio of
  the executed (squashed) action."""
  mu0, s0, mu1, s1 = torch.randn(16), torch.rand(16) + 0.3, torch.randn(16), torch.rand(16) + 0.3
  u = mu0 + s0 * torch.randn(16)
  a = torch.tanh(u)
  jac = torch.log(1 - a.square() + 1e-12).sum()
  n0, n1 = torch.distributions.Normal(mu0, s0), torch.distributions.Normal(mu1, s1)
  log_pi0 = n0.log_prob(torch.atanh(a.clamp(-1 + 1e-7, 1 - 1e-7))).sum() - jac
  log_pi1 = n1.log_prob(torch.atanh(a.clamp(-1 + 1e-7, 1 - 1e-7))).sum() - jac
  assert torch.allclose(log_pi1 - log_pi0, n1.log_prob(u).sum() - n0.log_prob(u).sum(), atol=1e-3)


# ------------------------------------------------------------ features/actors
def test_features_identical_for_symmetric_rotations():
  from in_hand_rotation_mjlab.estimation.rotations import cube_symmetry_matrices, quat_to_matrix
  from in_hand_rotation_mjlab.scratch_study.features import cube_state_features

  R = quat_to_matrix(random_quats(50).float())
  S = cube_symmetry_matrices()
  pos, v, w, c = torch.zeros(50, 3), torch.zeros(50, 3), torch.zeros(50, 3), torch.zeros(50, 4)
  f0 = cube_state_features(pos, R, v, w, c)
  for k in range(24):
    fk = cube_state_features(pos, R @ S[k], v, w, c)
    assert torch.allclose(f0, fk, atol=1e-5)


def _obs(n=8):
  return TensorDict({"actor": torch.randn(n, 320), "critic": torch.randn(n, 83), "cube_state": torch.randn(n, 17)}, batch_size=[n])


def _kwargs():
  return dict(hidden_dims=(64, 64), activation="elu", obs_normalization=True, stochastic=True, init_noise_std=0.7)


def test_privileged_information_isolation(tmp_path):
  from rsl_rl.models.mlp_model import MLPModel

  from in_hand_rotation_mjlab.estimation.model import CubeStateEstimator, save_estimator
  from in_hand_rotation_mjlab.scratch_study.actors import EstimatedStateMLPModel, OracleStateMLPModel

  groups = {"actor": ["actor"], "critic": ["critic"]}
  obs = _obs()
  est = CubeStateEstimator()
  save_estimator(est, tmp_path / "e.pt")
  A = MLPModel(obs, groups, "actor", 16, **_kwargs()).eval()
  C = EstimatedStateMLPModel(obs, groups, "actor", 16, estimator_path=str(tmp_path / "e.pt"), **_kwargs()).eval()
  D = OracleStateMLPModel(obs, groups, "actor", 16, **_kwargs()).eval()
  for m in (A, C, D):
    torch.nn.init.normal_(m.mlp[-1].weight, std=0.1)
  assert A.obs_dim == C.obs_dim == D.obs_dim == 320
  assert C.mlp[0].in_features == D.mlp[0].in_features == 339
  changed = obs.clone()
  changed["critic"] = torch.randn(8, 83)
  changed["cube_state"] = torch.randn(8, 17)
  with torch.no_grad():
    assert torch.equal(A(obs), A(changed))
    assert torch.equal(C(obs), C(changed))  # C sees only obs_320 (via the estimator)
    assert not torch.equal(D(obs), D(changed))  # D reads the oracle group
    only_critic = obs.clone()
    only_critic["critic"] = torch.randn(8, 83)
    assert torch.equal(D(obs), D(only_critic))  # ...but never the critic group
  assert not any(p.requires_grad for p in C.estimator.parameters())


def test_feature_ablation_overrides():
  from in_hand_rotation_mjlab.scratch_study.actors import OracleStateMLPModel

  obs = _obs()
  D = OracleStateMLPModel(obs, {"actor": ["actor"], "critic": ["critic"]}, "actor", 16, **_kwargs()).eval()
  torch.nn.init.normal_(D.mlp[-1].weight, std=0.1)
  with torch.no_grad():
    base = D(obs)
    D.feature_override = "zero"
    z = D(obs)
    D.feature_override = "shuffle"
    s = D(obs)
    D.feature_mean = torch.zeros(19)
    D.feature_override = "mean"
    m = D(obs)
  assert not torch.equal(base, z) and not torch.equal(base, s)
  assert torch.allclose(z, m)


def test_zero_actor_output_targets_policy_head_not_estimator(tmp_path):
  """Regression: train.py --zero-actor-output zeroes the last nn.Linear of the actor.

  It must hit the policy MLP's output layer for A, C and D, and must never touch
  C's frozen estimator (which would make the estimated features constant).
  """
  import importlib.util
  from pathlib import Path

  from rsl_rl.models.mlp_model import MLPModel

  from in_hand_rotation_mjlab.estimation.model import CubeStateEstimator, save_estimator
  from in_hand_rotation_mjlab.scratch_study.actors import EstimatedStateMLPModel, OracleStateMLPModel

  spec = importlib.util.spec_from_file_location("train_script", Path(__file__).parents[1] / "scripts" / "train.py")
  train_script = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(train_script)

  groups = {"actor": ["actor"], "critic": ["critic"]}
  obs = _obs()
  est = CubeStateEstimator()
  for p in est.parameters():
    torch.nn.init.normal_(p, std=0.05)
  save_estimator(est, tmp_path / "e.pt")
  ref = {k: v.clone() for k, v in est.state_dict().items()}
  actors = {
    "A": MLPModel(obs, groups, "actor", 16, **_kwargs()),
    "C": EstimatedStateMLPModel(obs, groups, "actor", 16, estimator_path=str(tmp_path / "e.pt"), **_kwargs()),
    "D": OracleStateMLPModel(obs, groups, "actor", 16, **_kwargs()),
  }
  for name, actor in actors.items():
    runner = SimpleNamespace(alg=SimpleNamespace(get_policy=lambda a=actor: a))
    train_script._zero_actor_output(runner)
    assert torch.count_nonzero(actor.mlp[-1].weight) == 0, name
    assert torch.count_nonzero(actor.mlp[-1].bias) == 0, name
  C = actors["C"].eval()
  for k, v in C.estimator.state_dict().items():
    assert torch.equal(v, ref[k]), f"estimator parameter {k} was modified"
  assert not any(k.startswith("estimator.") for k in C.state_dict())
  assert all(p.requires_grad is False for p in C.estimator.parameters())
  assert not any(p is q for p in C.parameters() for q in C.estimator.parameters())
  with torch.no_grad():
    C(obs)
  f = C.last_features
  assert f.shape == (8, 19)
  assert float(f.std(0).max()) > 1e-3, "estimated features are constant across inputs"
