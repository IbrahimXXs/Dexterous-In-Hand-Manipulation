"""Tests of C5-HW-V2 (experiments/c5_hw_v2, task Mjlab-Leap-C5HW2): it differs from C5-HW-EFFORT by exactly
the step-A response ranges and the step-B kit-start resets, each in its own config function."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

E = Path("experiments/c5_hw_v2")


def _diff(a, b, path="cfg", out=None):
  """Every leaf path where two config trees differ (same helper as tests/test_c5_hw_v1.py)."""
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
    except Exception:  # noqa: BLE001
      eq = bool(np.array_equal(np.asarray(a), np.asarray(b)))
    if not eq:
      out.append(path)
  return out


RESPONSE_LEAVES = ["cfg.events['dr_action_delay'].params['lag_range'][0]", "cfg.events['dr_action_delay'].params['lag_range'][1]",
                   "cfg.events['dr_motor_damping'].params['ranges'][0]", "cfg.events['dr_motor_damping'].params['ranges'][1]",
                   "cfg.events['dr_motor_friction'].params['ranges'][0]", "cfg.events['dr_motor_friction'].params['ranges'][1]"]
RESET_LEAVES = ["cfg.actions['joint_pos'].required_grasp_cache", "cfg.events['reset_from_grasp_cache'].params['cache_file']"]


def _fit():
  return json.loads((E / "results" / "response_fit.json").read_text())


def test_response_frozen_from_the_fit_or_dropped_by_its_rule():
  from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_c5_hw_v2.env_cfgs import RESPONSE

  fit = _fit()
  if fit["primary"]["status"] != "ok":
    assert RESPONSE is None
    return
  f = fit["fitted_randomisation"]
  assert RESPONSE == {"lag_range": tuple(f["lag_range_steps"]), "damping": tuple(f["damping_scale_range"]),
                      "friction": tuple(f["friction_scale_range"])}


def test_each_change_alone_and_together_differs_from_effort_exactly_as_documented():
  from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_c5_hw_v1.env_cfgs import c5hw_effort_env_cfg
  from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_c5_hw_v2 import env_cfgs as V2

  resp = [] if V2.RESPONSE is None else RESPONSE_LEAVES
  for play in (False, True):
    base = c5hw_effort_env_cfg(play=play)
    assert _diff(base, V2.apply_response(c5hw_effort_env_cfg(play=play))) == resp
    assert sorted(_diff(base, V2.apply_kit_start_resets(c5hw_effort_env_cfg(play=play)))) == RESET_LEAVES
    assert sorted(_diff(base, V2.c5hw2_env_cfg(play=play))) == sorted(resp + RESET_LEAVES)
  cfg = V2.c5hw2_env_cfg()
  assert cfg.events["reset_from_grasp_cache"].params["cache_file"] == str(V2.KIT_START_CACHE)
  assert cfg.events["reset_from_grasp_cache"].params["restore_robot_joint_state"] is True
  assert np.allclose(cfg.events["dr_effort_limits"].params["effort_limit"], (0.16695, 0.20405))
  assert cfg.rewards["actuator_utilisation_penalty"].weight == -0.2201
  assert cfg.rewards["rotate_finite_diff"].params["clip_max"] == 0.5


def test_registered_task_and_ppo_config():
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  import in_hand_rotation_mjlab.tasks  # noqa: F401

  ef, v2 = load_env_cfg("Mjlab-Leap-C5HW-EFFORT"), load_env_cfg("Mjlab-Leap-C5HW2")
  from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_c5_hw_v2.env_cfgs import RESPONSE
  resp = [] if RESPONSE is None else RESPONSE_LEAVES
  assert sorted(_diff(ef, v2)) == sorted(resp + RESET_LEAVES)
  pe, p2 = load_rl_cfg("Mjlab-Leap-C5HW-EFFORT"), load_rl_cfg("Mjlab-Leap-C5HW2")
  assert _diff(pe, p2) == ["cfg.experiment_name"] and p2.experiment_name == "c5_hw_v2"
  assert p2.max_iterations == 3001


def test_kit_start_cache_matches_its_record_and_kit_reset_semantics():
  from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_c5_hw_v2.env_cfgs import KIT_START_CACHE

  rec = json.loads(KIT_START_CACHE.with_suffix(".json").read_text())
  assert hashlib.sha256(KIT_START_CACHE.read_bytes()).hexdigest() == rec["cache_sha256"]
  c = np.load(KIT_START_CACHE)
  n = len(c["cube_size"])
  assert n == rec["kept"] and n > 1000
  for k in ("joint_pos", "joint_pos_target"):
    assert c[k].shape == (n, 16)
  assert c["cube_pose_rel"].shape == (n, 7) and np.isfinite(c["cube_pose_rel"]).all()
  # stored command = measured post-ramp pose (clamped into the approved limits), as the kit resets its previous command
  assert np.abs(c["joint_pos_target"] - c["joint_pos"]).max() <= rec["target_clamped_max_rad"] + 1e-6
  g = np.load("experiments/hwlimit_ac/cache/hwlimit_grasp_cache.npz")
  assert list(c["joint_names"]) == list(g["joint_names"])
  assert np.allclose(c["cube_size"], g["cube_size"][49])
  d = np.linalg.norm(c["cube_pose_rel"][:, :2] - g["cube_pose_rel"][49, :2], axis=1)
  assert d.max() <= 0.025 + 1e-6  # the pre-registered position filter


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@gpu
def test_episodes_start_from_the_kit_cache_with_the_measured_pose_as_command():
  from in_hand_rotation_mjlab.grasp_rotation_v2 import rollout as R
  from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_c5_hw_v2.env_cfgs import KIT_START_CACHE

  c = np.load(KIT_START_CACHE)
  env, raw, _ = R.make_env("Mjlab-Leap-C5HW2", 32, 4601, "cuda:0", num_steps=5)
  raw._gr2_assignment = torch.arange(32, device="cuda:0")
  env.reset()
  ids = raw._hand_cube_reset_cache_ids.cpu().numpy()
  assert (ids == np.arange(32)).all()
  robot = raw.scene["robot"]
  term = raw.action_manager.get_term("joint_pos")
  assert np.allclose(robot.data.joint_pos.cpu().numpy(), c["joint_pos"][ids], atol=1e-5)
  assert np.allclose(term._target.cpu().numpy(), c["joint_pos_target"][ids], atol=1e-5)
  lim = raw.sim.model.actuator_forcerange[:, robot.indexing.ctrl_ids, 1]
  assert float(lim.min()) >= 0.16695 - 1e-6 and float(lim.max()) <= 0.20405 + 1e-6
  env.close()
