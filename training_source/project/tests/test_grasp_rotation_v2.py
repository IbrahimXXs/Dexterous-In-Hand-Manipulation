"""Tests of the Grasp-Rotation-V2 experiment (experiments/grasp_rotation_v2)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from in_hand_rotation_mjlab.grasp_rotation_v2.rewards import retention_v2_raw
from in_hand_rotation_mjlab.retention_v1.rewards import PALM_MU, PALM_SD, retention_excess


# ---------------------------------------------------------------- reward maths (CPU)
def test_v2_penalty_equals_retention_v1_at_defaults_and_splits_exactly():
  g = torch.Generator().manual_seed(0)
  pos = torch.tensor(PALM_MU) + 4 * torch.randn(5000, 3, generator=g) * torch.tensor(PALM_SD)
  tilt = 1.2 * torch.rand(5000, generator=g)
  assert torch.allclose(retention_v2_raw(pos, tilt), retention_excess(pos, tilt), atol=1e-6)
  pos_only = retention_v2_raw(pos, tilt, tilt_scale_factor=0.0)
  tilt_only = retention_v2_raw(pos, tilt, pos_scale=0.0)
  assert torch.allclose(pos_only, retention_excess(pos, torch.zeros_like(tilt)), atol=1e-6)
  assert torch.allclose(tilt_only, retention_excess(torch.tensor(PALM_MU).expand(5000, 3), tilt), atol=1e-5)
  assert torch.allclose(pos_only + tilt_only, retention_excess(pos, tilt), atol=1e-5)
  assert (tilt_only[tilt <= 0.5] == 0).all() and (tilt_only >= 0).all()


def test_candidates_differ_from_retention_v1_by_one_documented_change():
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  import in_hand_rotation_mjlab.tasks  # noqa: F401
  from in_hand_rotation_mjlab.grasp_rotation_v2.rewards import retention_penalty_v2

  r = load_env_cfg("Mjlab-Leap-RetV1-A")
  pr = load_rl_cfg("Mjlab-Leap-RetV1-A")
  for cand in ("P0", "C5"):
    c = load_env_cfg(f"Mjlab-Leap-GRV2-{cand}")
    assert list(c.rewards) == list(r.rewards)
    changed = [k for k in r.rewards if c.rewards[k].func != r.rewards[k].func or c.rewards[k].params != r.rewards[k].params
               or c.rewards[k].weight != r.rewards[k].weight]
    if cand == "P0":
      assert changed == ["retention_penalty"]
      t = c.rewards["retention_penalty"]
      assert t.func is retention_penalty_v2 and t.weight == -0.05
      assert (t.params["pos_scale"], t.params["tilt_scale_factor"], t.params["tilt_margin"]) == (0.0, 1.0, 0.5)
    else:
      assert changed == ["rotate_finite_diff"]
      p0, p1 = r.rewards["rotate_finite_diff"].params, c.rewards["rotate_finite_diff"].params
      assert {k for k in p0 if p0[k] != p1[k]} == {"clip_min", "clip_max"} and (p1["clip_min"], p1["clip_max"]) == (-0.5, 0.5)
    assert list(c.terminations) == list(r.terminations) and list(c.events) == list(r.events)
    assert all(c.events[k].params == r.events[k].params for k in r.events)
    assert list(c.curriculum) == list(r.curriculum) and c.episode_length_s == r.episode_length_s == 20.0
    assert c.decimation == r.decimation and c.sim.mujoco.timestep == r.sim.mujoco.timestep
    assert type(c.actions["joint_pos"]) is type(r.actions["joint_pos"])
    assert list(c.observations["actor"].terms) == ["joint_pos", "prev_commanded_joint_pos"]
    pc = load_rl_cfg(f"Mjlab-Leap-GRV2-{cand}")
    assert pc.experiment_name == f"grasp_rotation_v2_{cand}"
    assert pc.algorithm == pr.algorithm and pc.actor == pr.actor and pc.critic == pr.critic
    assert pc.max_iterations == pr.max_iterations == 3001 and pc.obs_groups == pr.obs_groups and pc.clip_actions is None


# ---------------------------------------------------------------- environment (GPU)
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@gpu
def test_assigned_reset_pins_grasps_and_matches_cache():
  from in_hand_rotation_mjlab.grasp_rotation_v2.rollout import rollout

  a = (np.arange(64) * 7) % 239
  tr = rollout("Mjlab-Leap-RetV1-A", None, 4401, 64, 5, "cuda:0", assignment=a)
  c = np.load("experiments/hwlimit_ac/cache/hwlimit_grasp_cache.npz")
  assert (tr["cache_id"] == a).all()
  assert np.abs(tr["init_command"] - c["joint_pos_target"][a]).max() < 1e-6
  # without an assignment the draw is the task's own (uniform, several grasps)
  tr2 = rollout("Mjlab-Leap-RetV1-A", None, 4402, 64, 2, "cuda:0")
  assert len(np.unique(tr2["cache_id"])) > 20


@gpu
@pytest.mark.parametrize("cand", ["P0", "C5"])
def test_candidate_rewards_match_offline_recomputation(cand):
  from in_hand_rotation_mjlab.grasp_rotation_v2.rollout import rollout

  tr = rollout(f"Mjlab-Leap-GRV2-{cand}", None, 4403, 256, 60, "cuda:0", assignment=np.arange(256) % 239)
  names = [str(x) for x in tr["reward_names"]]
  R = tr["reward"].astype(np.float32)
  held = tr["alive"].astype(bool)
  held[:10] = False
  pos, tilt = torch.from_numpy(tr["pos_palm"].astype(np.float32)), torch.from_numpy(tr["tilt"].astype(np.float32))
  if cand == "P0":
    want = -0.05 * retention_v2_raw(pos, tilt, pos_scale=0.0).numpy() * 0.05
  else:
    want = -0.05 * retention_excess(pos, tilt).numpy() * 0.05
  assert np.abs(R[..., names.index("retention_penalty")] - want)[held].max() < 5e-3
  assert {"grasp_loss", "retention_penalty", "rotate_finite_diff"} <= set(names)
