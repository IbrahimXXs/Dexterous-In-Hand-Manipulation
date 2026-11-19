"""Tests of the Retention-V1 experiment (experiments/retention_v1)."""

from __future__ import annotations

import pytest
import torch

from in_hand_rotation_mjlab.retention_v1.rewards import CAP, MARGIN_SD, PALM_MU, PALM_SD, retention_excess

MU = torch.tensor(PALM_MU)
SD = torch.tensor(PALM_SD)


def _p(offsets_sd):
  return (MU + torch.tensor(offsets_sd, dtype=torch.float32) * SD)[None]


# ---------------------------------------------------------------- reward maths (CPU)
def test_retention_zero_inside_region_and_on_safe_sides():
  z = torch.zeros(1)
  assert retention_excess(_p([0, 0, 0]), z).item() == 0.0
  for d in range(3):
    for s in (-1, 1):
      off = [0.0, 0.0, 0.0]
      off[d] = s * (MARGIN_SD - 0.01)
      assert retention_excess(_p(off), z).item() == 0.0
  # safe directions (-x, -y, +z) are never penalised, however far
  assert retention_excess(_p([-6, -6, 6]), z).item() == 0.0
  assert retention_excess(_p([0, 0, 0]), torch.tensor([0.5])).item() == 0.0


def test_retention_quadratic_and_capped_on_risky_sides():
  z = torch.zeros(1)
  for d, s in ((0, 1), (1, 1), (2, -1)):
    off = [0.0, 0.0, 0.0]
    off[d] = s * (MARGIN_SD + 1.0)
    assert retention_excess(_p(off), z).item() == pytest.approx(1.0, rel=1e-4)
    off[d] = s * (MARGIN_SD + 2.0)
    assert retention_excess(_p(off), z).item() == pytest.approx(4.0, rel=1e-4)
    off[d] = s * (MARGIN_SD + 10.0)
    assert retention_excess(_p(off), z).item() == pytest.approx(CAP ** 2, rel=1e-4)
  t = torch.tensor([0.6, 0.7, 0.8, 1.5])
  assert torch.allclose(retention_excess(MU.expand(4, 3).clone(), t), torch.tensor([1.0, 4.0, 9.0, 9.0]), atol=1e-4)
  # components add
  assert retention_excess(_p([MARGIN_SD + 1, MARGIN_SD + 1, 0]), torch.tensor([0.6])).item() == pytest.approx(3.0, rel=1e-4)


def test_configs_differ_from_hwlimit_A_only_by_the_two_reward_terms():
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  import in_hand_rotation_mjlab.tasks  # noqa: F401

  a, r = load_env_cfg("Mjlab-Leap-HwLim-A"), load_env_cfg("Mjlab-Leap-RetV1-A")
  assert set(r.rewards) - set(a.rewards) == {"retention_penalty", "grasp_loss"} and set(a.rewards) <= set(r.rewards)
  for k in a.rewards:
    assert r.rewards[k].func == a.rewards[k].func and r.rewards[k].weight == a.rewards[k].weight
  assert r.rewards["retention_penalty"].weight == -0.05 and r.rewards["grasp_loss"].weight == -10.0
  assert list(r.terminations) == list(a.terminations) and list(r.events) == list(a.events)
  assert list(r.curriculum) == list(a.curriculum) and r.episode_length_s == a.episode_length_s == 20.0
  assert r.decimation == a.decimation and r.sim.mujoco.timestep == a.sim.mujoco.timestep
  assert type(r.actions["joint_pos"]) is type(a.actions["joint_pos"])
  assert r.actions["joint_pos"].required_grasp_cache == a.actions["joint_pos"].required_grasp_cache
  assert list(r.observations["actor"].terms) == list(a.observations["actor"].terms) == ["joint_pos", "prev_commanded_joint_pos"]
  assert r.events["reset_from_grasp_cache"].params["cache_file"] == a.events["reset_from_grasp_cache"].params["cache_file"]
  pa, pr = load_rl_cfg("Mjlab-Leap-HwLim-A"), load_rl_cfg("Mjlab-Leap-RetV1-A")
  assert pr.experiment_name == "retention_v1_A" and pa.experiment_name == "hwlim_A"
  assert pr.algorithm == pa.algorithm and pr.actor == pa.actor and pr.critic == pa.critic
  assert pr.max_iterations == pa.max_iterations == 3001 and pr.clip_actions is None and pr.obs_groups == pa.obs_groups


# ---------------------------------------------------------------- environment (GPU)
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _env(n, seed, episode_length_s=None):
  from in_hand_rotation_mjlab.hwlimit_study.evaluation import make_env

  return make_env("Mjlab-Leap-RetV1-A", n, seed, "cuda:0", episode_length_s=episode_length_s)


@gpu
def test_actor_observation_and_limits_unchanged():
  from in_hand_rotation_mjlab.hwlimit_study.limits import kit_limits_sim_order

  env, raw, agent = _env(16, 4301)
  om = raw.observation_manager
  assert om.active_terms["actor"] == ["joint_pos", "prev_commanded_joint_pos"] and om.group_obs_dim["actor"] == (320,)
  term = raw.action_manager.get_term("joint_pos")
  lo, hi = term.approved_limits()
  klo, khi = (torch.as_tensor(x, device="cuda:0", dtype=torch.float32) for x in kit_limits_sim_order())
  assert torch.all(lo >= klo - 1e-6) and torch.all(hi <= khi + 1e-6)
  env.reset()
  for t in range(40):
    env.step(4.0 * torch.randn(16, 16, device="cuda:0"))
    assert torch.all(term._target >= lo - 1e-6) and torch.all(term._target <= hi + 1e-6)
  env.close()


@gpu
def test_grasp_loss_once_per_pose_failure_never_on_timeout():
  env, raw, _ = _env(64, 4302, episode_length_s=2.0)
  rm, tm = raw.reward_manager, raw.termination_manager
  gl = rm._term_names.index("grasp_loss")
  env.reset()
  g = torch.Generator(device="cuda:0").manual_seed(0)
  n_pose = n_to = n_g = 0
  for t in range(200):  # random actions: many pose failures and many timeouts
    env.step(3.0 * torch.randn(64, 16, device="cuda:0", generator=g))
    pose = tm.get_term("cube_pose_deviation").bool() & ~tm.get_term("cube_fell").bool()
    fired = rm._step_reward[:, gl] != 0
    assert torch.equal(fired, pose)
    assert not bool((fired & tm.time_outs.bool() & ~pose).any())
    assert torch.allclose(rm._step_reward[fired, gl], torch.full_like(rm._step_reward[fired, gl], -10.0))
    n_pose += int(pose.sum()); n_to += int((tm.time_outs.bool() & ~pose).sum()); n_g += int(fired.sum())
  assert n_pose > 0 and n_to > 0 and n_g == n_pose
  env.close()


@gpu
def test_retention_reward_matches_independent_computation():
  """The retention term uses exactly the termination's tilt reference (same anchoring rule),
  and equals ``retention_excess`` of the palm-frame cube position and that tilt.  Rewards
  are computed before MJLab's final ``sim.forward()`` of the step, so positions lag the
  observation by one physics substep (5 ms; shared by every existing reward term); the
  comparison therefore uses held commands, skips the settling steps right after a reset
  and allows 5e-3."""
  env, raw, _ = _env(64, 4303)
  rm, tm = raw.reward_manager, raw.termination_manager
  k = rm._term_names.index("retention_penalty")
  ret = rm.get_term_cfg("retention_penalty").func
  term = tm.get_term_cfg("cube_pose_deviation").func
  env.reset()
  since = torch.zeros(64, dtype=torch.long, device="cuda:0")
  nonzero = 0
  for t in range(60):
    obs, _, dones, _ = env.step(torch.zeros(64, 16, device="cuda:0"))
    since = torch.where(dones.bool(), torch.zeros_like(since), since + 1)
    assert torch.equal(ret.anchor.up0, term.anchor.up0) and torch.equal(ret.anchor.pos0, term.anchor.pos0)
    ok = since >= 10
    if not bool(ok.any()):
      continue
    expect = retention_excess(obs["cube_state"][:, 0:3], term.anchor.tilt()) * -0.05
    got = rm._step_reward[:, k]
    assert torch.allclose(got[ok], expect[ok], atol=5e-3), (got[ok] - expect[ok]).abs().max()
    nonzero += int((expect[ok] != 0).sum())
  assert nonzero > 0  # the comparison exercised the penalised region
  env.close()
