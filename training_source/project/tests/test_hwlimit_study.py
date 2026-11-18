"""Tests of the hardware-limit A/C study (experiments/hwlimit_ac)."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
KIT = REPO / "LEAP_Hand_RL_Deploy_Kit" / "kit" / "deploy_direct.py"
E = ROOT / "experiments" / "hwlimit_ac"
CACHE = E / "cache" / "hwlimit_grasp_cache.npz"


def _kit_array(name: str) -> np.ndarray:
  tree = ast.parse(KIT.read_text())
  for node in tree.body:
    if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
      return np.array(ast.literal_eval(node.value.args[0]), dtype=np.float64)
  raise KeyError(name)


# ---------------------------------------------------------------- CPU tests
def test_limits_equal_deploy_kit():
  from in_hand_rotation_mjlab.hwlimit_study import limits as L

  assert np.array_equal(_kit_array("_HW_LIMIT_LO"), np.array(L.KIT_HW_LIMIT_LO))
  assert np.array_equal(_kit_array("_HW_LIMIT_HI"), np.array(L.KIT_HW_LIMIT_HI))
  assert np.array_equal(_kit_array("_HW_TO_SERVER").astype(int), np.array(L.KIT_HW_TO_SERVER))
  src = KIT.read_text()
  assert '"--limit-margin-rad", type=float, default=0.01' in src
  env = (REPO / "LEAP_Hand_RL_Deploy_Kit" / "configs" / "hand.env").read_text()
  assert "MAX_COMMAND_STEP_RAD=0.05" in env


def test_limits_in_simulation_order():
  from in_hand_rotation_mjlab.hwlimit_study.limits import JOINT_NAMES, kit_limits_sim_order

  lo, hi = kit_limits_sim_order()
  d = dict(zip(JOINT_NAMES, zip(lo, hi)))
  for f in ("if", "mf", "rf"):
    assert d[f"{f}_rot"] == pytest.approx((-0.513599, 0.513599))
    for j in ("mcp", "pip", "dip"):
      assert d[f"{f}_{j}"] == pytest.approx((-0.251799, 1.90986))
  assert d["th_cmc"] == pytest.approx((0.01, 1.561)) and d["th_axl"] == pytest.approx((0.01, 1.561))
  assert d["th_mcp"] == pytest.approx((-0.339066, 1.73533)) and d["th_ipl"] == pytest.approx((-0.339066, 1.73533))


def test_ppo_configs_identical_except_actor_and_names():
  from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_hwlimit_study.rl_cfg import ESTIMATOR_B, hwlimit_ppo_cfg
  from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_scratch_study.rl_cfg import scratch_ppo_cfg

  a, c = hwlimit_ppo_cfg("A"), hwlimit_ppo_cfg("C")
  assert a.max_iterations == c.max_iterations == 3001 and a.save_interval == c.save_interval == 50
  assert (a.experiment_name, c.experiment_name) == ("hwlim_A", "hwlim_C")
  assert a.algorithm == c.algorithm == scratch_ppo_cfg("A").algorithm
  assert a.critic == c.critic and a.clip_actions is None and c.clip_actions is None
  for k in ("hidden_dims", "activation", "obs_normalization", "init_noise_std", "noise_std_type", "stochastic"):
    assert getattr(a.actor, k) == getattr(c.actor, k)
  assert Path(c.actor.estimator_path) == ESTIMATOR_B and ESTIMATOR_B.is_file()
  with pytest.raises(ValueError):
    hwlimit_ppo_cfg("D")


def test_estimator_b_unchanged():
  line = (E / "preservation" / "estimator_B.sha256").read_text().split()
  assert hashlib.sha256((ROOT / line[1]).read_bytes()).hexdigest() == line[0] == (
    "6e333d6067dc0b5c64dafee29180bf2c7ccde581cca146ae548a1f6d5fe13e64")


def test_grasp_cache_inside_approved_limits():
  from in_hand_rotation_mjlab.hwlimit_study.limits import JOINT_NAMES, kit_limits_sim_order

  d = np.load(CACHE)
  assert tuple(str(x) for x in d["joint_names"]) == JOINT_NAMES
  lo, hi = kit_limits_sim_order()
  assert np.all(d["joint_pos_target"] >= lo - 1e-6) and np.all(d["joint_pos_target"] <= hi + 1e-6)
  excess = np.maximum(lo - d["joint_pos"], d["joint_pos"] - hi).clip(min=0)
  assert excess.max() <= 0.02 + 1e-6
  assert len(d["cube_size"]) >= 100 and d["fingertip_contact"].sum(1).min() >= 1
  line = (CACHE.with_suffix(".sha256")).read_text().split()
  assert hashlib.sha256(CACHE.read_bytes()).hexdigest() == line[0]


def test_original_study_task_unchanged():
  from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_scratch_study.actions import SquashedJointPositionDeltaActionCfg
  from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_scratch_study.env_cfgs import scratch_env_cfg
  from mjlab.tasks.registry import load_env_cfg

  import in_hand_rotation_mjlab.tasks  # noqa: F401

  cfg = load_env_cfg("Mjlab-Leap-Scratch-A")
  assert type(cfg.actions["joint_pos"]) is SquashedJointPositionDeltaActionCfg
  assert cfg.events["reset_from_grasp_cache"].params["cache_file"].endswith("leap_right_mbzuai_58mm_policy_grasp_cache.npz")
  assert type(scratch_env_cfg().actions["joint_pos"]) is SquashedJointPositionDeltaActionCfg


# ---------------------------------------------------------------- GPU tests
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@gpu
def test_commands_stay_inside_approved_limits_and_match_observation():
  from in_hand_rotation_mjlab.hwlimit_study.evaluation import make_env
  from in_hand_rotation_mjlab.hwlimit_study.limits import KIT_MAX_COMMAND_STEP_RAD, kit_limits_sim_order

  env, raw, agent_cfg = make_env("Mjlab-Leap-HwLim-A", 32, 4243, "cuda:0", obs_noise_off=True)
  assert agent_cfg.clip_actions is None
  assert raw.observation_manager.active_terms["actor"] == ["joint_pos", "prev_commanded_joint_pos"]
  term = raw.action_manager.get_term("joint_pos")
  lo, hi = term.approved_limits()
  klo, khi = (torch.as_tensor(x, device="cuda:0", dtype=torch.float32) for x in kit_limits_sim_order())
  assert torch.all(lo >= klo - 1e-6) and torch.all(hi <= khi + 1e-6)
  env.reset()
  obs = env.get_observations()
  assert obs["actor"].shape[-1] == 320
  g = torch.Generator(device="cuda:0").manual_seed(0)
  prev = term._target.clone()
  hit = torch.zeros(16, dtype=torch.bool, device="cuda:0")
  for t in range(120):
    u = 3.0 * torch.randn(32, 16, device="cuda:0", generator=g) + (4.0 if t < 60 else -4.0)  # drive into both limits
    obs, _, dones, _ = env.step(u)
    cmd = term._target
    assert torch.all(cmd >= lo - 1e-6) and torch.all(cmd <= hi + 1e-6)
    ok = ~dones.bool()
    assert torch.all((cmd - prev)[ok].abs() <= KIT_MAX_COMMAND_STEP_RAD)
    # newest commanded-history frame (obs[160+144:320], absolute) equals the clamped command
    assert torch.allclose(obs["actor"][ok, 304:320], cmd[ok], atol=1e-5)
    hit |= ((cmd - lo).abs() < 1e-5).any(0) | ((hi - cmd).abs() < 1e-5).any(0)
    prev = cmd.clone()
  assert bool(hit.all()), "every joint should reach an approved limit under saturating actions"
  env.close()


@gpu
def test_reset_commands_inside_limits():
  from in_hand_rotation_mjlab.hwlimit_study.evaluation import make_env

  env, raw, _ = make_env("Mjlab-Leap-HwLim-C", 64, 4244, "cuda:0")
  term = raw.action_manager.get_term("joint_pos")
  lo, hi = term.approved_limits()
  env.reset()
  assert torch.all(term._target >= lo - 1e-6) and torch.all(term._target <= hi + 1e-6)
  assert raw._hand_cube_grasp_cache_path.endswith("hwlimit_grasp_cache.npz")
  env.close()
