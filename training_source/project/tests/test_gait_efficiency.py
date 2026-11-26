"""Tests for the gait-efficiency cycle counter (experiments/gait_efficiency/scripts/gait.py) and the simulation
analysis path of sim_gait.py on synthetic data (CPU only, no simulator)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

E = Path(__file__).resolve().parents[1] / "experiments" / "gait_efficiency"


def _load(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[name] = mod
  spec.loader.exec_module(mod)
  return mod


G = _load("gait_test", E / "scripts" / "gait.py")


def _strokes(f_hz=0.3, T_s=60.0, fs=20.0, amp=0.4, noise=0.01, seed=0):
  rng = np.random.default_rng(seed)
  t = np.arange(int(T_s * fs)) / fs
  v = rng.normal(size=16)
  v /= np.linalg.norm(v)
  q = 1.0 + np.outer(amp * np.sin(2 * np.pi * f_hz * t), v) + noise * rng.normal(size=(len(t), 16))
  return q, t


def test_counts_cycles_and_rhythm_of_a_sinusoid():
  q, t = _strokes()
  c = G.count_cycles(q, t)
  assert c["status"] == "ok"
  assert c["cycles"] == pytest.approx(0.3 * (t[-1] - 3.0), rel=0.03)
  assert c["rhythm_hz"] == pytest.approx(0.3, abs=0.01)
  assert c["stroke_amplitude_median_rad"] == pytest.approx(0.8, rel=0.1)  # peak-to-peak of a 0.4-rad sine
  assert c["pc1_variance_share"] > 0.9


def test_independent_of_joint_order_and_sign():
  q, t = _strokes(seed=3)
  perm = np.random.default_rng(1).permutation(16)
  sign = np.where(np.arange(16) % 3 == 0, -1.0, 1.0)
  a, b = G.count_cycles(q, t), G.count_cycles(q[:, perm] * sign, t)
  assert a["cycles"] == pytest.approx(b["cycles"], rel=1e-6)
  assert a["rhythm_hz"] == pytest.approx(b["rhythm_hz"], abs=1e-9)


def test_window_end_and_short_window():
  q, t = _strokes()
  half = G.count_cycles(q, t, t_end_s=33.0)
  assert half["cycles"] == pytest.approx(0.3 * 30.0, rel=0.05)
  assert G.count_cycles(q, t, t_end_s=4.0)["status"] == "window too short"


def test_sim_analysis_on_synthetic_rollout(tmp_path, monkeypatch):
  S = _load("sim_gait_test", E / "scripts" / "sim_gait.py")
  monkeypatch.setattr(S, "E", tmp_path)
  (tmp_path / "data").mkdir()
  T, n, dt = 1200, 3, 0.05
  t = np.arange(T) * dt
  q = np.stack([_strokes(seed=s)[0] for s in range(n)], axis=1).astype(np.float32)  # (T, n, 16)
  yaw = np.zeros((T, n), np.float32)
  yaw[1:, :] = (2 * np.pi / 4) * 0.25 / 20.0  # 0.25 quarter turns per second while held
  end = np.array([-1, 400, 100])  # held 60 s, dropped at 20 s, dropped at 5 s
  for e in range(n):
    if end[e] >= 0:
      yaw[end[e]:, e] = 0.0
  np.savez(tmp_path / "data" / "x.npz", q=q, yaw_inc=yaw, end_step=end, end_code=np.zeros(n), cache_id=np.arange(n), dt=np.float32(dt),
           joint_names=np.array(["j"] * 16))
  rows = S.per_env("x")
  assert [round(r["held_s"], 2) for r in rows] == [60.0, 20.0, 5.0]
  assert rows[0]["qturns"] == pytest.approx(0.25 * 59.95, rel=1e-3)
  assert rows[0]["cycles"] == pytest.approx(0.3 * (59.95 - 3.0), rel=0.03)
  s = S.summarise(rows, np.random.default_rng(0))
  assert s["n_envs"] == 2  # held >= 20 s
  # quarter turns over the whole held time (as the operator counts), cycles from 3 s on
  expected = (0.25 * 59.95 + 0.25 * 19.95) / (0.3 * (59.95 - 3.0) + 0.3 * (19.95 - 3.0))
  assert s["qturns_per_cycle"] == pytest.approx(expected, rel=0.04)
  w = S.summarise(rows, np.random.default_rng(0), qkey="qturns_window")
  assert w["qturns_per_cycle"] == pytest.approx(0.25 / 0.3, rel=0.04)
  del t
