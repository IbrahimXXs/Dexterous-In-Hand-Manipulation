"""CPU tests for the seed-jury study (experiments/seed_jury): jury rules and the replay window helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

E = Path(__file__).resolve().parents[1] / "experiments" / "seed_jury" / "scripts"


def _load(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[name] = mod
  spec.loader.exec_module(mod)
  return mod


SJ = _load("seed_jury_sim", E / "sim_jury.py")


def _actors(a):
  return [lambda obs, x=x: x.clone() for x in a]


def test_median_is_the_per_joint_median_command_and_has_zero_disagreement():
  g = torch.Generator().manual_seed(0)
  a = torch.randn(3, 5, 16, generator=g) * 2
  J = SJ.JuryActor(_actors(a), "J-median")
  u = J(None)
  assert torch.allclose(torch.tanh(u), torch.tanh(a).median(0).values)
  assert float(J.D[0].abs().max()) < 1e-7


def test_mean_executes_the_mean_delta():
  g = torch.Generator().manual_seed(1)
  a = torch.randn(3, 4, 16, generator=g)
  u = SJ.JuryActor(_actors(a), "J-mean")(None)
  assert torch.allclose(torch.tanh(u) * SJ.DELTA, (torch.tanh(a) * SJ.DELTA).mean(0), atol=1e-6)


def test_veto_replaces_only_joints_beyond_tau_and_singles_pass_through():
  a = torch.zeros(3, 1, 16)
  a[0, 0, :] = 0.1
  a[1, 0, :] = 0.2
  a[2, 0, :] = 0.15  # C5s3 = median everywhere ...
  a[2, 0, 3] = 3.0   # ... except joint 3, far out
  tau = np.full(16, 0.01)
  u = SJ.JuryActor(_actors(a), "J-veto", tau)(None)
  assert torch.allclose(u[0, :3], a[2, 0, :3]) and float(u[0, 3]) == pytest.approx(0.2)
  for i, s in enumerate(SJ.SEED_ORDER):
    J = SJ.JuryActor(_actors(a), s)
    assert torch.equal(J(None), a[i])


def test_calibration_logs_c5s3_deviation():
  a = torch.zeros(3, 2, 16)
  a[2] = 1.0
  J = SJ.JuryActor(_actors(a), "C5s3", calibrate=True)
  J(None)
  assert torch.allclose(J.dev3[0], torch.full((2, 16), float(np.tanh(1.0)) * SJ.DELTA))


def test_item_list_and_replay_windows():
  its = SJ.items()
  assert its[0]["calibration"] and len(its) == 1 + 5 * 6 * 2
  RR = _load("seed_jury_real_replay_t", E / "real_replay.py")
  x = np.arange(20, dtype=float)
  tr = RR.trailing(x, 10)
  assert tr[0] == 0 and tr[9] == pytest.approx(4.5) and tr[19] == pytest.approx(14.5)
  assert RR.control_windows({}, 10.0) == [(3.0, 5.0), (5.0, 7.0), (7.0, 9.0)]
  assert RR.auc([2.0, 3.0], [1.0, 2.0]) == pytest.approx(0.875)
