"""CPU tests for the PT-student study (experiments/pt_student): tokens, students, targets, loss, teacher task."""

from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

S = Path(__file__).resolve().parents[1] / "experiments" / "pt_student" / "scripts"


def _load(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[name] = mod
  spec.loader.exec_module(mod)
  return mod


M = _load("pt_models_t", S / "models.py")


def test_tokens_follow_the_term_major_layout():
  obs = torch.arange(320, dtype=torch.float32)[None]
  tok = M.to_tokens(obs)
  assert tok.shape == (1, 10, 32)
  assert torch.equal(tok[0, 3, :16], torch.arange(48, 64, dtype=torch.float32))  # joint positions, frame 3
  assert torch.equal(tok[0, 3, 16:], torch.arange(160 + 48, 160 + 64, dtype=torch.float32))  # previous commands, frame 3


@pytest.mark.parametrize("cls", [M.TransformerStudent, M.MLPStudent])
def test_students_shapes_zero_action_and_roundtrip(cls, tmp_path):
  torch.manual_seed(0)
  m = cls()
  x = torch.randn(64, 320)
  m.norm.fit(x)
  out = m(x)
  assert {k: tuple(v.shape) for k, v in out.items()} == {"action": (64, 16), "pos": (64, 3), "ori": (64, 4), "joint": (64, 16)}
  assert float(out["action"].abs().max()) == 0.0  # zero-initialised action head
  M.save(m, tmp_path / "m.pt")
  m2 = M.load(tmp_path / "m.pt")
  for k in out:
    assert torch.allclose(m2(x)[k], out[k], atol=1e-6)


def test_transformer_size_is_as_planned():
  m = M.TransformerStudent()
  assert m.cfg == {"kind": "transformer", "d": 256, "layers": 3, "heads": 4, "ff": 1024}
  assert len(m.encoder.layers) == 3 and m.encoder.layers[0].self_attn.num_heads == 4
  assert m.encoder.layers[0].linear1.out_features == 1024


def _quat_z(a):
  return torch.tensor([[math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]])


def test_orientation_target_is_yaw_mod_90_and_tilt_free_when_flat():
  pytest.importorskip("in_hand_rotation_mjlab")
  base = torch.zeros(1, 17)
  crit = torch.zeros(1, 83)
  outs = []
  for a in (0.3, 0.3 + math.pi / 2, 0.3 + math.pi):
    cs = base.clone()
    cs[:, 3:7] = _quat_z(a)
    outs.append(M.aux_targets(cs, crit)["ori"])
  for o in outs[1:]:
    assert torch.allclose(o, outs[0], atol=1e-5)  # cube symmetry: 90 deg turns give the same target
  assert float(M.yaw_from_ori(outs[0])) == pytest.approx(0.3, abs=1e-5)
  assert torch.allclose(outs[0][0, 2:], torch.zeros(2), atol=1e-6)  # flat cube: no tilt


def test_loss_is_zero_for_a_perfect_student():
  out = {"action": torch.zeros(4, 16), "pos": torch.ones(4, 3), "ori": torch.zeros(4, 4), "joint": torch.ones(4, 16)}
  lt = M.loss_terms(out, torch.zeros(4, 16), {"pos": torch.ones(4, 3), "ori": torch.zeros(4, 4), "joint": torch.ones(4, 16)})
  assert all(float(v) == 0.0 for v in lt.values())


def test_teacher_task_differs_from_c5_only_in_what_the_actor_sees():
  pytest.importorskip("mjlab")
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  from in_hand_rotation_mjlab.estimation.sim import _import_tasks

  _import_tasks()
  assert repr(load_env_cfg("Mjlab-Leap-PT-Teacher")) == repr(load_env_cfg("Mjlab-Leap-GRV2-C5"))
  t, c = asdict(load_rl_cfg("Mjlab-Leap-PT-Teacher")), asdict(load_rl_cfg("Mjlab-Leap-GRV2-C5"))
  diff = {k for k in t if t[k] != c[k]}
  assert diff == {"experiment_name", "obs_groups"}, diff
  assert tuple(t["obs_groups"]["actor"]) == ("actor", "critic") and tuple(t["obs_groups"]["critic"]) == ("critic",)
  assert t["max_iterations"] == 3001


def test_action_rate_teacher_adds_only_the_squashed_action_rate_term():
  pytest.importorskip("mjlab")
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  from in_hand_rotation_mjlab.estimation.sim import _import_tasks

  _import_tasks()
  ar, c5 = load_env_cfg("Mjlab-Leap-PT-Teacher-AR"), load_env_cfg("Mjlab-Leap-GRV2-C5")
  extra = ar.rewards.pop("action_rate_squashed")
  assert extra.weight == -0.075 and extra.func.__name__ == "squashed_action_rate_l2"
  assert repr(ar) == repr(c5)  # nothing else differs
  t, c = asdict(load_rl_cfg("Mjlab-Leap-PT-Teacher-AR")), asdict(load_rl_cfg("Mjlab-Leap-PT-Teacher"))
  assert {k for k in t if t[k] != c[k]} == {"experiment_name"} and t["experiment_name"] == "pt_student_teacher_ar"
  assert "action_rate_squashed" not in load_env_cfg("Mjlab-Leap-PT-Teacher").rewards


def test_squashed_action_rate_is_the_change_in_the_executed_action():
  pytest.importorskip("in_hand_rotation_mjlab")
  from types import SimpleNamespace

  from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_pt_student.rewards import squashed_action_rate_l2

  u, p = torch.tensor([[0.5, -2.0, 0.0]]), torch.tensor([[0.5, 1.0, 3.0]])
  env = SimpleNamespace(action_manager=SimpleNamespace(action=u, prev_action=p))
  want = (torch.tanh(u) - torch.tanh(p)).square().sum(-1)
  assert torch.allclose(squashed_action_rate_l2(env), want)
  env.action_manager.prev_action = u.clone()
  assert float(squashed_action_rate_l2(env)) == 0.0
