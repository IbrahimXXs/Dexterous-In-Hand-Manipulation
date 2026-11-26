"""scripts/retime_recording.py: a recording tagged 30 fps but captured slower is re-timed to real time."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
cv2 = pytest.importorskip("cv2")


def test_retime_uses_frames_csv_clock(tmp_path: Path):
  rec = tmp_path / "recording"
  rec.mkdir()
  T0 = 1_800_000_000.0
  # 40 frames captured at an irregular ~10 fps, but the file is tagged 30 fps (the 25 Nov recorder behaviour)
  rng = np.random.default_rng(0)
  times = T0 + np.cumsum(np.r_[0.0, rng.uniform(0.08, 0.12, 39)])
  vw = cv2.VideoWriter(str(rec / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (416, 32))
  for k in range(40):
    img = np.zeros((32, 416, 3), np.uint8)
    img[:, 8 + 10 * k:16 + 10 * k] = 255  # frame k: a white bar at x = 8 + 10k
    vw.write(img)
  vw.release()
  with open(rec / "frames.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame", "host_time_s", "camera_time_s"])
    for k, t in enumerate(times):
      w.writerow([k, f"{t:.6f}", ""])
  # kit log whose policy starts 1.5 s after the first frame
  started = datetime.fromtimestamp(T0 + 1.0, tz=timezone.utc).isoformat()
  np.savez(tmp_path / "run.npz", started_at=np.array(started), elapsed_s=np.array([0.5, 0.55]))

  out = tmp_path / "rt.mp4"
  subprocess.run([sys.executable, str(ROOT / "scripts" / "retime_recording.py"), str(rec), "--log",
                  str(tmp_path / "run.npz"), "--fps", "20", "--out", str(out)], check=True, capture_output=True)
  info = json.loads(out.with_suffix(".json").read_text())
  assert info["tagged_fps"] == pytest.approx(30.0)
  assert info["measured_capture_fps"] == pytest.approx(39 / (times[-1] - times[0]), rel=1e-3)
  assert info["policy_start_in_out_s"] == pytest.approx(1.5, abs=1e-3)

  def frame_ids(path: Path) -> np.ndarray:
    cap, ids = cv2.VideoCapture(str(path)), []
    while True:
      ok, img = cap.read()
      if not ok:
        return np.asarray(ids)
      ids.append(int(round((img.mean(axis=(0, 2)).argmax() - 8) / 10)))

  assert cv2.VideoCapture(str(out)).get(cv2.CAP_PROP_FPS) == pytest.approx(20.0)
  levels = frame_ids(out)
  # output duration = capture duration (real time), not 40 frames / 30 fps
  assert len(levels) == info["out_frames"] == len(np.arange(times[0], times[-1] + 1e-9, 1 / 20))
  # each output tick shows the latest frame captured at or before it (frame index decoded from the bar position)
  ticks = times[0] + np.arange(len(levels)) / 20
  want = np.searchsorted(times, ticks, side="right") - 1
  assert np.array_equal(levels, want)
