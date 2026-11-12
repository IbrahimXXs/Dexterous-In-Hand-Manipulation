"""GPU tests of the corrected study environment (Mjlab-Leap-Scratch-*).

    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl python -m pytest -q tests/test_scratch_env.py
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = "cuda:0"


@pytest.fixture(scope="module")
def sim():
  from in_hand_rotation_mjlab.estimation.sim import make_env

  env, raw, agent_cfg = make_env(64, 321, DEV, task="Mjlab-Leap-Scratch-A", obs_corruption=False)
  yield env, raw
  env.close()


def test_observation_groups(sim):
  env, raw = sim
  obs = env.get_observations()
  assert obs["actor"].shape[-1] == 320
  assert obs["critic"].shape[-1] == 83
  assert obs["cube_state"].shape[-1] == 17
  om = raw.observation_manager
  assert list(om.active_terms["actor"]) == ["joint_pos", "prev_commanded_joint_pos"]


def test_no_spurious_termination_after_resets(sim):
  env, raw = sim
  zero = torch.zeros(raw.num_envs, 16, device=DEV)
  term = raw.termination_manager
  # Repeated full resets from arbitrary previous states.  The anchoring step
  # can never terminate.  A few grasp-cache spawns genuinely topple onto
  # another face (~90 deg physical tilt) within 0.15 s (measured 2 / 2048);
  # the stale-anchor bug terminated most episodes immediately.
  early = 0
  for _ in range(3):
    env.reset()
    for _ in range(25):
      env.step(zero)
    env.reset()
    env.step(zero)
    assert int(term.get_term("cube_pose_deviation").sum()) == 0
    for _ in range(2):
      env.step(zero)
      early += int(term.get_term("cube_pose_deviation").sum())
  assert early <= 0.01 * 3 * raw.num_envs


def test_cube_state_group_matches_critic_ground_truth(sim):
  from in_hand_rotation_mjlab.estimation.sim import GroundTruth

  env, raw = sim
  env.reset()
  obs, *_ = env.step(torch.zeros(raw.num_envs, 16, device=DEV))
  gt = GroundTruth(raw)
  lab = gt.labels(obs["critic"])
  cs = obs["cube_state"]
  assert torch.allclose(cs[:, 0:3], lab["pos"], atol=1e-6)
  assert torch.allclose(cs[:, 7:10], lab["lin_vel"], atol=1e-6)
  assert torch.allclose(cs[:, 10:13], lab["ang_vel"], atol=1e-6)
  assert torch.equal(cs[:, 13:17], lab["contact"].float())


def test_env_executes_tanh_of_policy_output(sim):
  env, raw = sim
  env.reset()
  u = torch.randn(raw.num_envs, 16, device=DEV) * 2.0
  term = raw.action_manager.get_term("joint_pos")
  before = term._target.clone()
  env.step(u)
  assert torch.allclose(term.executed_normalized, torch.tanh(u))
  lo, hi = term._joint_limits(slice(None))
  expected = (before + torch.tanh(u) * (1 / 24)).clamp(lo, hi)
  assert torch.allclose(term._target, expected, atol=1e-6)


def test_cube_drop_terminates(sim):
  env, raw = sim
  env.reset()
  zero = torch.zeros(raw.num_envs, 16, device=DEV)
  env.step(zero)
  cube = raw.scene["cube"]
  pose = torch.cat([cube.data.root_link_pos_w.clone(), cube.data.root_link_quat_w.clone()], -1)
  ids = torch.arange(0, raw.num_envs, 2, device=DEV)
  pose[ids, 2] = 0.05  # far below the palm / 0.2 m drop threshold
  cube.write_root_link_pose_to_sim(pose[ids], env_ids=ids)
  _, _, dones, _ = env.step(zero)
  fell = raw.termination_manager.get_term("cube_fell")
  assert bool(fell[ids].all())
  assert bool(dones[ids].all())
