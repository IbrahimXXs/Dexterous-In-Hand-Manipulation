#!/usr/bin/env python3
"""Re-time a cube_vision recording to real time, for scripts/make_hero_gif.sh.

The recorder (training_source/project/real_world/cube_vision/sources.py, ``Recorder``) tags rgb.mp4 with a
nominal frame rate, but the camera loop may run slower (25 Nov trial 1: about 11 fps in a file tagged 30 fps,
so a plain replay runs about 2.7x too fast). The true capture time of every frame is in frames.csv
(``frame,host_time_s,camera_time_s``). This script writes a constant-rate copy in which each output tick
shows the latest frame captured at or before that tick, so seconds in the output are real seconds.

It also prints where the policy started inside the output (from the kit run log), so the GIF window can be
given as "seconds after the policy start".

Usage (repo root, vision extra):
  uv run --extra vision python scripts/retime_recording.py \\
      outputs/cube_vision/live_20261125_111603/recording \\
      --log LEAP_Hand_RL_Deploy_Kit/logs/physical_20261125_111626.npz \\
      --out outputs/hero_gif/trial1_realtime.mp4
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def read_host_times(frames_csv: Path) -> np.ndarray:
  rows = {}
  with open(frames_csv, newline="") as f:
    for r in csv.DictReader(f):
      rows[int(r["frame"])] = float(r["host_time_s"])
  n = max(rows) + 1 if rows else 0
  t = np.full(n, np.nan)
  for k, v in rows.items():
    t[k] = v
  return t


def tick_indices(t: np.ndarray, fps: float) -> tuple[np.ndarray, float]:
  """For each output tick, the index of the latest frame captured at or before it."""
  ok = np.where(np.isfinite(t))[0]
  if len(ok) < 2:
    raise SystemExit("frames.csv has fewer than two timed frames")
  tt = t[ok]
  if np.any(np.diff(tt) < 0):
    raise SystemExit("frames.csv host times are not monotonic")
  ticks = np.arange(tt[0], tt[-1] + 1e-9, 1.0 / fps)
  j = np.searchsorted(tt, ticks, side="right") - 1
  return ok[np.clip(j, 0, len(ok) - 1)], float(tt[0])


def policy_start_host(log: Path) -> float:
  d = np.load(log, allow_pickle=False)
  return datetime.fromisoformat(str(d["started_at"])).timestamp() + float(d["elapsed_s"][0])


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("recording", type=Path, help="folder with rgb.mp4 and frames.csv")
  ap.add_argument("--log", type=Path, help="kit run log (.npz) of the same trial: prints the policy start offset")
  ap.add_argument("--fps", type=float, default=30.0, help="output frame rate (real time)")
  ap.add_argument("--out", type=Path, required=True)
  a = ap.parse_args()

  import cv2

  t = read_host_times(a.recording / "frames.csv")
  idx, t_first = tick_indices(t, a.fps)
  cap = cv2.VideoCapture(str(a.recording / "rgb.mp4"))
  tagged = cap.get(cv2.CAP_PROP_FPS)
  n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  ok_t = t[np.isfinite(t)]
  measured = (len(ok_t) - 1) / (ok_t[-1] - ok_t[0])
  a.out.parent.mkdir(parents=True, exist_ok=True)
  w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  vw = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
  k, frame = -1, None
  for want in idx:
    while k < want:
      okr, img = cap.read()
      if not okr:
        break
      k, frame = k + 1, img
    if frame is None:
      raise SystemExit("could not read rgb.mp4")
    vw.write(frame)
  vw.release()
  info = {"recording": str(a.recording), "out": str(a.out), "tagged_fps": round(tagged, 2),
          "measured_capture_fps": round(float(measured), 2), "video_frames": n_video,
          "timed_frames": int(np.isfinite(t).sum()), "out_fps": a.fps, "out_frames": int(len(idx)),
          "first_frame_host_s": t_first, "duration_s": round(float(ok_t[-1] - ok_t[0]), 3)}
  if a.log:
    p0 = policy_start_host(a.log)
    info["policy_start_host_s"] = p0
    info["policy_start_in_out_s"] = round(p0 - t_first, 3)
  print(json.dumps(info, indent=1))
  a.out.with_suffix(".json").write_text(json.dumps(info, indent=1))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
