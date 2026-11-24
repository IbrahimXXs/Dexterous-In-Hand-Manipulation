"""Tests of C5-HW-V1 condition C5-HW-effort (experiments/c5_hw_v1, task Mjlab-Leap-C5HW-EFFORT)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from in_hand_rotation_mjlab.c5_hw_v1.rewards import utilisation_hinge

E = Path("experiments/c5_hw_v1")


def _diff(a, b, path="cfg", out=None):
  """Every leaf path where two config trees differ (dataclasses, dicts, lists, tuples, leaves by ==)."""
  out = [] if out is None else out
  if type(a) is not type(b):
    out.append(path)
  elif dataclasses.is_dataclass(a) and not isinstance(a, type):
    for f in dataclasses.fields(a):
      _diff(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}", out)
  elif isinstance(a, dict):
    for k in sorted(set(a) | set(b), key=str):
      if k not in a or k not in b:
        out.append(f"{path}[{k!r}]")
      else:
        _diff(a[k], b[k], f"{path}[{k!r}]", out)
  elif isinstance(a, (list, tuple)):
    if len(a) != len(b):
      out.append(path)
    else:
      for i, (x, y) in enumerate(zip(a, b)):
        _diff(x, y, f"{path}[{i}]", out)
  else:
    try:
      eq = bool(a == b)
    except Exception:  # noqa: BLE001 (tensors / arrays)
      eq = bool(np.array_equal(np.asarray(a), np.asarray(b)))
    if not eq:
      out.append(path)
  return out


# ---------------------------------------------------------------- reward maths (CPU)
def test_hinge_zero_below_threshold_one_at_limit_smooth_and_per_actuator():
  u = torch.linspace(0, 1, 1001)[:, None]
  h = utilisation_hinge(u)
  assert (h[u[:, 0] <= 0.8] == 0).all()
  assert torch.isclose(h[-1], torch.tensor(1.0))
  assert torch.isclose(utilisation_hinge(torch.tensor([[0.9]])), torch.tensor(0.25))
  assert (torch.diff(h) >= 0).all()  # monotone
  d = torch.diff(h) / 1e-3
  assert d[:800].abs().max() == 0 and d[800] < 0.05 and d.max() <= 10.0 + 1e-3  # C1 at 0.8; slope 10 at the limit
  v = torch.tensor([[0.9, 1.0, 0.5, 0.85]])
  assert torch.isclose(utilisation_hinge(v), torch.tensor(0.25 + 1.0 + 0.0 + 0.0625))


def test_effort_range_is_the_deployed_350mA_in_the_simulators_mapping():
  from in_hand_rotation_mjlab.robots.leap_hand.leap_right_constants import (
    LEAP_ACTUATOR_EFFORT_LIMIT_NM, LEAP_MOTOR_CURRENT_LIMIT_MA, LEAP_MOTOR_TORQUE_PER_AMP_NM)
  from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_c5_hw_v1.env_cfgs import deployed_effort_range

  assert (LEAP_MOTOR_TORQUE_PER_AMP_NM, LEAP_MOTOR_CURRENT_LIMIT_MA) == (0.53, 550.0)
  base = (0.9 * LEAP_ACTUATOR_EFFORT_LIMIT_NM, 1.1 * LEAP_ACTUATOR_EFFORT_LIMIT_NM)
  lo, hi = deployed_effort_range(base)
  assert np.allclose((lo, hi), (0.9 * 0.53 * 0.35, 1.1 * 0.53 * 0.35)) and np.allclose((lo, hi), (0.16695, 0.20405))
  kit = Path("../../LEAP_Hand_RL_Deploy_Kit/configs/hand.env")
  assert "CURRENT_LIMIT_MA=350" in kit.read_text().splitlines()


def test_weight_frozen_from_calibration():
  from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_c5_hw_v1.env_cfgs import EFFORT_PENALTY_WEIGHT

  cal = json.loads((E / "results" / "effort_weight_calibration.json").read_text())
  assert EFFORT_PENALTY_WEIGHT == cal["w_rounded_frozen"] == 0.2201
  assert abs(cal["w"] * cal["penalty_p_per_step_mean"] - 0.1 * cal["rotation_reward_per_step_mean"]) < 1e-9


def test_effort_condition_differs_from_c5_only_by_the_two_documented_changes():
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  import in_hand_rotation_mjlab.tasks  # noqa: F401
  from in_hand_rotation_mjlab.c5_hw_v1.rewards import actuator_utilisation_penalty

  for play in (False, True):
    from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_c5_hw_v1.env_cfgs import c5hw_effort_env_cfg
    from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_grasp_rotation_v2.env_cfgs import grv2_env_cfg
    c5, ef = grv2_env_cfg("C5", play=play), c5hw_effort_env_cfg(play=play)
    d = _diff(c5, ef)
    assert d == ["cfg.events['dr_effort_limits'].params['effort_limit'][0]",
                 "cfg.events['dr_effort_limits'].params['effort_limit'][1]",
                 "cfg.rewards['actuator_utilisation_penalty']"], d
  c5, ef = load_env_cfg("Mjlab-Leap-GRV2-C5"), load_env_cfg("Mjlab-Leap-C5HW-EFFORT")
  assert np.allclose(ef.events["dr_effort_limits"].params["effort_limit"], (0.16695, 0.20405))
  t = ef.rewards["actuator_utilisation_penalty"]
  assert t.func is actuator_utilisation_penalty and t.weight == -0.2201
  assert "actuator_utilisation_penalty" not in {v.params.get("reward_name") for v in ef.curriculum.values()}
  assert ef.rewards["rotate_finite_diff"].params["clip_max"] == 0.5 and ef.rewards["rotate_finite_diff"].params["clip_min"] == -0.5
  assert list(ef.observations["actor"].terms) == ["joint_pos", "prev_commanded_joint_pos"]
  pc, pe = load_rl_cfg("Mjlab-Leap-GRV2-C5"), load_rl_cfg("Mjlab-Leap-C5HW-EFFORT")
  assert _diff(pc, pe) == ["cfg.experiment_name"] and pe.experiment_name == "c5_hw_v1_EFFORT"
  assert pe.max_iterations == 3001


# ---------------------------------------------------------------- environment (GPU)
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@gpu
def test_capped_limits_applied_and_penalty_matches_offline_recomputation():
  from in_hand_rotation_mjlab.grasp_rotation_v2 import rollout as R

  env, raw, _ = R.make_env("Mjlab-Leap-C5HW-EFFORT", 64, 4501, "cuda:0", num_steps=40)
  robot = raw.scene["robot"]
  ctrl = robot.indexing.ctrl_ids
  env.reset()
  rm = raw.reward_manager
  ip = list(rm._term_names).index("actuator_utilisation_penalty")
  lim = raw.sim.model.actuator_forcerange[:, ctrl, 1]
  assert lim.shape == (64, 16) and float(lim.min()) >= 0.16695 - 1e-6 and float(lim.max()) <= 0.20405 + 1e-6
  assert float(lim.std()) > 0.005  # randomised independently per environment and actuator
  assert torch.allclose(raw.sim.model.actuator_forcerange[:, ctrl, 0], -lim)
  # capture the actuator force and limits the term sees at reward time (before resets / the next forward)
  from in_hand_rotation_mjlab.c5_hw_v1 import rewards as CR
  seen_inputs = {}
  cfg = rm._term_cfgs[ip]
  orig = cfg.func

  def probe(env_, **kw):
    seen_inputs["F"] = robot.data.actuator_force.clone()
    seen_inputs["L"] = raw.sim.model.actuator_forcerange[:, ctrl, 1].clone()
    return orig(env_, **kw)

  cfg.func = probe
  g = torch.Generator(device="cuda:0").manual_seed(0)
  seen = 0.0
  for _ in range(40):
    env.step(2.0 * torch.rand((64, 16), device="cuda:0", generator=g) - 1.0)
    F, L = seen_inputs["F"], seen_inputs["L"]
    assert (F.abs() <= L + 1e-5).all()
    p = torch.square(torch.clamp(F.abs() / L - 0.8, min=0) / 0.2).sum(-1)
    assert torch.allclose(rm._step_reward[:, ip], -0.2201 * p, atol=1e-5)
    seen += float(p.sum())
  assert CR.UTIL_THRESHOLD == 0.8
  assert seen > 0  # random actions reach the limits
  env.close()


@gpu
def test_original_c5_environment_still_uses_the_550mA_range():
  from in_hand_rotation_mjlab.grasp_rotation_v2 import rollout as R

  env, raw, _ = R.make_env("Mjlab-Leap-GRV2-C5", 32, 4502, "cuda:0", num_steps=5)
  env.reset()
  lim = raw.sim.model.actuator_forcerange[:, raw.scene["robot"].indexing.ctrl_ids, 1]
  assert float(lim.min()) >= 0.26235 - 1e-6 and float(lim.max()) <= 0.32065 + 1e-6
  assert "actuator_utilisation_penalty" not in raw.reward_manager._term_names
  env.close()
