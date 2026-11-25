"""scripts/vision_trials.py: per-trial camera rotation from dashboard poses + kit run logs."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("cv2")  # cube_vision needs the vision extra


def _rot(axis: np.ndarray, ang: float) -> np.ndarray:
  a = axis / np.linalg.norm(axis)
  K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
  return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def test_vision_quarter_turns_per_trial(tmp_path: Path):
  axis = np.array([0.3, -0.2, 0.93])
  T0 = 1_800_000_000.0
  # trials: (start offset s, duration s, rate rad/s, drop time s or None)
  trials = [(5.0, 20.0, +1.2, None), (40.0, 20.0, -0.6, 8.0), (70.0, 10.0, +2.0, None)]
  rate_at = lambda tt: next((rate for s, dur, rate, _ in trials if s <= tt - T0 <= s + dur), 0.0)  # noqa: E731
  ang, frames, t_prev = 0.0, [], T0
  for k in range(int(90 * 30)):
    tt = T0 + k / 30
    ang += rate_at(tt) * (tt - t_prev)
    t_prev = tt
    hidden = 50 <= tt - T0 <= 50.3  # short occlusion inside trial 2 (bridged)
    frames.append({"host_time_s": tt, "tracking_valid": not hidden, "pose_ambiguous": False,
                   "cube_rotation_matrix": _rot(axis, ang).tolist()})
  vis = tmp_path / "live_x"
  vis.mkdir()
  (vis / "poses.jsonl").write_text("\n".join(json.dumps(f) for f in frames) + "\n")
  rows = []
  for i, (s, dur, rate, drop) in enumerate(trials, 1):
    log = tmp_path / f"physical_{i}.npz"
    np.savez(log, started_at=np.array(datetime.fromtimestamp(T0 + s - 1.0, timezone.utc).isoformat()),
             elapsed_s=np.linspace(1.0, 1.0 + dur, int(dur * 20)))
    rows.append({"trial": str(i), "policy": "C5s3", "log": log.name, "observed_turns": "", "dropped": str(drop is not None),
                 "drop_time_s": "" if drop is None else str(drop)})
  tr = tmp_path / "trials_test.csv"
  with open(tr, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
  subprocess.run([sys.executable, str(ROOT / "scripts" / "vision_trials.py"), "--trials", str(tr), "--vision", str(vis),
                  "--axis-sign-from-trial", "1"], check=True, capture_output=True)
  out = list(csv.DictReader(open(tmp_path / "trials_test_vision.csv")))
  for (s, dur, rate, drop), r in zip(trials, out):
    assert float(r["vision_quarter_turns"]) == pytest.approx(rate * dur / (math.pi / 2), abs=0.15)
    assert float(r["vision_coverage"]) > 0.95
    if drop is not None:
      assert float(r["vision_qt_until_drop"]) == pytest.approx(rate * drop / (math.pi / 2), abs=0.15)


def test_gapfill_long_gap_needs_speed_not_shortest_path():
  """2.5 s hidden at 1.5 rad/s = 3.75 rad (215 deg): the shortest path says -145 deg (wrong sign);
  the speed-predicted branch recovers +215 deg, the old rule drops the interval."""
  sys.path.insert(0, str(ROOT / "training_source" / "project" / "real_world"))
  from cube_vision.gapfill import GapConfig, integrate_yaw
  axis = np.array([0.2, 0.1, 0.97])
  t = np.arange(0, 8, 1 / 30)
  ang = 1.5 * t
  Rs = [None if 3.0 < ti < 5.5 else _rot(axis, a) for ti, a in zip(t, ang)]
  cfg = GapConfig(accel_sd=0.5)
  r = integrate_yaw(t, Rs, axis, cfg)
  assert r.total_rad == pytest.approx(ang[-1] - ang[0], abs=0.05)
  assert [g.status for g in r.gaps] == ["resolved"]


def test_gapfill_flags_unpredictable_gap_and_twist_ignores_swing():
  sys.path.insert(0, str(ROOT / "training_source" / "project" / "real_world"))
  from cube_vision.gapfill import GapConfig, integrate_yaw, twist_angle
  axis = np.array([0.0, 0.0, 1.0])
  # pure swing about x does not count as rotation about z
  assert abs(twist_angle(_rot(np.array([1.0, 0, 0]), 0.4), axis)) < 1e-9
  assert twist_angle(_rot(np.array([1.0, 0, 0]), 0.3) @ _rot(axis, 0.7), axis) == pytest.approx(0.7, abs=0.05)
  # speed reverses while hidden for 4 s -> the branch cannot be trusted -> flagged, range covers truth
  t = np.arange(0, 10, 1 / 30)
  rate = np.where(t < 5, 2.0, -2.0)
  ang = np.cumsum(rate) / 30
  Rs = [None if 3.0 < ti < 7.0 else _rot(axis, a) for ti, a in zip(t, ang)]
  r = integrate_yaw(t, Rs, axis, GapConfig(accel_sd=2.0))
  assert r.gaps[0].status == "ambiguous"
  assert r.lo_rad - 1e-6 <= ang[-1] - ang[0] + 2 * np.pi and r.hi_rad + 1e-6 >= ang[-1] - ang[0] - 2 * np.pi


def test_fill_estimates_hidden_frames_and_resolves_one_face_flip():
  """Hidden 1.2 s while spinning at 1.4 rad/s: the filled orientation follows the truth; a 1-face frame
  with two candidates (truth and a 70-deg flip) inside the gap is resolved to the truth."""
  sys.path.insert(0, str(ROOT / "training_source" / "project" / "real_world"))
  from cube_vision import fill as F
  from cube_vision.geometry import rotation_angle_deg
  axis = np.array([0.1, -0.8, -0.55])
  axis /= np.linalg.norm(axis)
  t = np.arange(0, 6, 1 / 30)
  ang = 1.4 * t + 0.2 * np.sin(1.3 * t)
  R0 = _rot(np.array([1.0, 0.3, 0.0]), 0.7)
  truth = [_rot(axis, a) @ R0 for a in ang]
  hidden = (t > 2.4) & (t < 3.6)
  Rs = [None if h else R for R, h in zip(truth, hidden)]
  P = [None if h else np.array([0.05, -0.03, 0.40]) for h in hidden]
  C = [None] * len(t)
  k = int(np.argmin(np.abs(t - 3.0)))
  C[k] = [(_rot(np.array([0.0, 1.0, 0.0]), np.radians(70)) @ truth[k], np.array([0.05, -0.03, 0.40])),
          (truth[k], np.array([0.05, -0.03, 0.40]))]
  ft = F.fill_track(t, Rs, P, C, axis)
  assert ft.status[k] == F.RESOLVED and rotation_angle_deg(ft.R[k], truth[k]) < 1.0
  est = [i for i in range(len(t)) if ft.status[i] == F.ESTIMATED]
  assert len(est) > 20
  assert max(rotation_angle_deg(ft.R[i], truth[i]) for i in est) < 10.0
  assert ft.yaw[-1] - ft.yaw[0] == pytest.approx(ang[-1] - ang[0], abs=0.05)
