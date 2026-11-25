#!/usr/bin/env python3
"""Per-trial camera video with the cube state filled in where the fingers hide it.

Re-tracks a trial recording (``outputs/cube_vision/trial_*``, written by ``./leap.sh trial`` with
LEAP_USE_VISION=1) with the region / range guards, then fills every frame (cube_vision/fill.py):

  MEASURED             pose seen (>= 2 faces, or 1 face resolved by the tracker)
  RESOLVED_1FACE       one face, its two-fold ambiguity resolved from the motion on both sides
  ESTIMATED            hidden: interpolated between the views before and after, with the number of
                       turns made while hidden chosen from the speed on both sides
  ESTIMATED_UNCERTAIN  hidden, and the number of turns while hidden is not certain (+-1 turn)
  NOT_VISIBLE          before the first / after the last view (e.g. after a drop): nothing estimated

Writes into <trial dir>/refined/:
  poses.jsonl            measured + resolved frames (used by scripts/vision_trials.py when present)
  frames_estimated.csv   per frame: time, status, quarter turns since the policy started, speed, gap
  summary.json
and with --render: <trial dir>/dashboard_estimated.mp4, the dashboard panel in REAL TIME (frames
placed by their timestamps), the rotation numbers and plots gap-filled, an orange dashed cube where
the pose is estimated, and a status bar.

    uv run --extra vision python scripts/trial_video.py outputs/cube_vision/trial_<...> \\
        --log LEAP_Hand_RL_Deploy_Kit/logs/physical_<...>.npz --layout ... --print-layout ... \\
        --roi 400,0,1280,720 --max-range-m 0.7 [--axis-json <trials>_vision_axis.json] [--render]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training_source" / "project" / "real_world"))
from cube_vision import app  # noqa: E402
from cube_vision import fill as F  # noqa: E402

QT = math.pi / 2
ORANGE, AMBER, GREEN, CYAN, GREY = (0, 140, 255), (0, 200, 255), (80, 220, 80), (230, 200, 60), (150, 150, 150)
STATUS_COL = {F.MEASURED: GREEN, F.RESOLVED: CYAN, F.ESTIMATED: ORANGE, F.UNCERTAIN: (60, 60, 240), F.NOT_VISIBLE: GREY}


def analyse(a):
  ns = argparse.Namespace(layout=a.layout, print_layout=a.print_layout, intrinsics=str(a.dir / "device.intrinsics.json"),
                          camera_to_palm=None, camera_yaw_axis=None, no_twin=True,
                          video=str(a.dir / "recording" / "rgb.mp4"),
                          depth=str(a.dir / "recording" / "depth") if (a.dir / "recording" / "depth").is_dir() else None,
                          depth_scale=None, timestamps=str(a.dir / "recording" / "frames.csv"),
                          log=str(a.log) if a.log else None, sync_offset_s=0.0, out=None, max_frames=None,
                          annotated_video=False, roi=a.roi, max_range_m=a.max_range_m)
  return app.analyse_recording(ns, progress=False)


def candidates_for(st, tracker):
  """Both IPPE solutions (refined) of an ambiguous single-face pose."""
  pose = st.track.pose
  if pose is None or not pose.ambiguous or len(pose.inlier_ids) != 1:
    return None
  det = next((d for d in st.track.detections if d.marker_id == pose.inlier_ids[0]), None)
  if det is None:
    return None
  out = []
  for R0, t0, _ in tracker._ippe(det):
    R, t = tracker._refine([det], R0, t0)
    out.append((R, np.asarray(t, np.float64).reshape(3)))
  return out


def position(st):
  if st.depth is not None and st.depth.position_m is not None:
    return np.asarray(st.depth.position_m, np.float64)
  return np.asarray(st.track.pose.t, np.float64).reshape(3)


def cube_corners(half):
  return np.array([[x, y, z] for x in (-half, half) for y in (-half, half) for z in (-half, half)], np.float64)


EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]


def draw_estimated(img, R, p, half, intr, col, label):
  import cv2

  pts = (R @ cube_corners(half).T).T + p
  uv, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), intr.camera_matrix, intr.dist_coeffs)
  uv = uv.reshape(-1, 2)
  if not np.isfinite(uv).all() or np.abs(uv).max() > 1e5:
    return
  for i, j in EDGES:
    a, b = uv[i], uv[j]
    for s in np.linspace(0, 1, 9)[::2]:  # dashed
      q0 = a + (b - a) * s
      q1 = a + (b - a) * min(s + 1 / 8, 1)
      cv2.line(img, tuple(int(v) for v in q0), tuple(int(v) for v in q1), col, 2, cv2.LINE_AA)
  # the spin axis through the centre
  c = uv.mean(axis=0)
  cv2.putText(img, label, (int(c[0] - 70), int(uv[:, 1].min() - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
  cv2.putText(img, label, (int(c[0] - 70), int(uv[:, 1].min() - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("dir", type=Path, help="trial camera folder (outputs/cube_vision/trial_...)")
  ap.add_argument("--log", type=Path, help="kit run log of this trial (.npz): sets the policy start (turn counter = 0)")
  ap.add_argument("--layout")
  ap.add_argument("--print-layout")
  ap.add_argument("--roi", help="x0,y0,x1,y1 marker detection region (pixels)")
  ap.add_argument("--max-range-m", type=float)
  ap.add_argument("--axis-json", type=Path, help="session axis written by vision_trials.py (same sign as the trial table)")
  ap.add_argument("--render", action="store_true", help="also write dashboard_estimated.mp4")
  ap.add_argument("--fps", type=float, default=20.0, help="output video rate (real time)")
  ap.add_argument("--before-s", type=float, default=5.0, help="video starts this long before the policy (cube placement)")
  ap.add_argument("--after-s", type=float, default=3.0, help="video ends this long after the run")
  a = ap.parse_args()

  A = analyse(a)
  states, tracker, intr, model = A["states"], A["pipe"].tracker, A["intr"], A["model"]
  if not states:
    sys.exit("no frames")
  t = np.array([s.track.host_time_s for s in states])
  if np.median(t) > 1e9:  # a frame without a frames.csv row (recorder stopped between the two writes) has no clock time
    keep = t > 1e9
    states, t = [s for s, k in zip(states, keep) if k], t[keep]
    A["pipe"].events.events[:] = [e for e in A["pipe"].events.events if -1e6 < e.t < 1e6]
  Rs = [s.track.pose.R if s.track.tracking_valid else None for s in states]
  P = [position(s) if s.track.tracking_valid else None for s in states]
  C = [candidates_for(s, tracker) for s in states]

  # policy window (turn counter starts at the policy start)
  w0, w1 = t[0], t[-1]
  if a.log:
    from vision_trials import run_window

    w0, w1 = run_window(a.log)
  axis, sign_note = None, "axis from this trial; sign = net direction of this trial"
  if a.axis_json and a.axis_json.exists():
    axis = np.asarray(json.loads(a.axis_json.read_text())["axis"], np.float64)
    sign_note = "session axis and sign (vision_trials.py)"
  if axis is None:  # from the motion during the policy only (not the operator placing the cube)
    from cube_vision.gapfill import dominant_axis

    m = np.where((t >= w0) & (t <= w1))[0]
    axis = dominant_axis(t[m], [Rs[i] for i in m]) if len(m) else None
    fixed_sign = False
  else:
    fixed_sign = True
  ft = F.fill_track(t, Rs, P, C, axis)
  if not fixed_sign:
    m = (t >= w0) & (t <= w1) & np.isfinite(ft.yaw)
    if m.sum() >= 2 and ft.yaw[m][-1] - ft.yaw[m][0] < 0:
      ft.yaw, ft.axis = -ft.yaw, -ft.axis
  # zero at the policy start (the operator places the cube before that)
  in_run = (t >= w0) & np.isfinite(ft.yaw)
  y0 = ft.yaw[np.argmax(in_run)] if in_run.any() else 0.0
  yq = (ft.yaw - y0) / QT
  yq[t < w0] = np.nan
  spd = np.degrees(F.speed_series(t, ft.yaw))

  out = a.dir / "refined"
  out.mkdir(exist_ok=True)
  with open(out / "poses.jsonl", "w") as f:
    for i, s in enumerate(states):
      ok = ft.status[i] in (F.MEASURED, F.RESOLVED)
      f.write(json.dumps({"frame": i, "host_time_s": float(t[i]), "tracking_valid": bool(ok), "pose_ambiguous": False,
                          "status": ft.status[i],
                          "cube_rotation_matrix": ft.R[i].tolist() if ok else None,
                          "cube_center_m": ft.p[i].tolist() if (ok and ft.p[i] is not None) else None}) + "\n")
  with open(out / "frames_estimated.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame", "host_time_s", "t_policy_s", "status", "quarter_turns", "speed_deg_s", "hidden_for_s", "gap_status",
                "gap_p_wrong"])
    for i in range(len(t)):
      g = ft.result.gaps[ft.gap_of[i]] if ft.gap_of[i] >= 0 else None
      w.writerow([i, f"{t[i]:.6f}", f"{t[i] - w0:.3f}", ft.status[i], "" if not np.isfinite(yq[i]) else f"{yq[i]:.3f}",
                  "" if not np.isfinite(spd[i]) else f"{spd[i]:.1f}", f"{ft.hidden_for[i]:.2f}" if g else "",
                  g.status if g else "", "" if (g is None or g.p_wrong is None) else f"{g.p_wrong:.4f}"])
  run = (t >= w0) & (t <= w1)
  counts = F.status_counts([s for s, r in zip(ft.status, run) if r])
  gs = [g for g in ft.result.gaps if g.t1 >= w0 and g.t0 <= w1]
  last = np.where(run & np.isfinite(yq))[0]
  summ = {"frames": len(t), "fps": round(len(t) / max(t[-1] - t[0], 1e-6), 1), "policy_window_frames": int(run.sum()),
          "status_fraction_in_run": counts, "resolved_1face_frames": ft.resolved_frames,
          "quarter_turns_run": None if not len(last) else round(float(yq[last[-1]]), 2),
          "gaps_in_run": len(gs), "gaps_uncertain": sum(g.status == "ambiguous" for g in gs),
          "uncertain_at_s": [[round(g.t0 - w0, 1), round(g.t1 - w0, 1)] for g in gs if g.status == "ambiguous"],
          "axis_camera": np.round(ft.axis, 4).tolist(), "axis_note": sign_note, "roi": a.roi, "max_range_m": a.max_range_m}
  (out / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
  pct = lambda k: f"{100 * counts[k]:.0f}%"  # noqa: E731
  print(f"[trial-video] {a.dir.name}: {summ['fps']} fps; during the policy: measured {pct(F.MEASURED)}, 1-face resolved "
        f"{pct(F.RESOLVED)}, estimated {pct(F.ESTIMATED)}, uncertain {pct(F.UNCERTAIN)}, not visible {pct(F.NOT_VISIBLE)}; "
        f"quarter turns {summ['quarter_turns_run']} ({summ['gaps_uncertain']} uncertain gap(s))")
  if not a.render:
    return 0

  # ---------------------------------------------------------------- render (real time)
  import cv2
  from cube_vision.dashboard import ROW1_H, ROW1_Y, Dashboard, ReplayIndex, ViewInfo, put

  for i, s in enumerate(states):  # the panel shows the filled values
    if i in ft.choice:
      R, p = ft.choice[i]
      s.track.pose.R, s.track.pose.t, s.track.pose.ambiguous = R, p, False
      s.track.pose.disambiguation = "both sides of the gap"
    s.yaw_rad = 0.0 if not np.isfinite(yq[i]) else float(yq[i] * QT)
    s.yaw_rate = None if not np.isfinite(spd[i]) else math.radians(float(spd[i]))
    s.revolutions = s.yaw_rad / (2 * math.pi)
    s.yaw_axis_source = "gap-filled (" + sign_note + ")"
  dash = Dashboard(model, intr, with_twin=False)
  idx = ReplayIndex(states)
  runlog = A["runlog"]
  info = ViewInfo("REPLAY", f"{a.dir.name}: {len(t)} frames, camera {summ['fps']} fps, shown in real time  [gap-filled]",
                  A["src"].has_depth,
                  ("telemetry: " + a.log.name) if a.log else "telemetry: none", A["sync"], n_frames=len(states),
                  stop_reason=runlog.stop_reason if runlog else None)
  src = A["src"]
  half = model.half_m if not callable(getattr(model, "half_m", None)) else model.half_m()
  p_out = a.dir / "dashboard_estimated.mp4"
  vw = cv2.VideoWriter(str(p_out), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (1680, 1000))
  cache = {}
  t_a, t_b = (max(t[0], w0 - a.before_s), min(t[-1], w1 + a.after_s)) if a.log else (t[0], t[-1])
  ticks = np.arange(t_a, t_b + 1e-9, 1.0 / a.fps)
  for tk in ticks:
    i = int(np.clip(np.searchsorted(t, tk, side="right") - 1, 0, len(t) - 1))
    if i not in cache:
      cache.clear()
      f = src.read_at(i)
      bgr = f.bgr.copy()
      stt = ft.status[i]
      if stt in (F.ESTIMATED, F.UNCERTAIN) and ft.R[i] is not None and ft.p[i] is not None:
        draw_estimated(bgr, ft.R[i], ft.p[i], half, intr, STATUS_COL[stt],
                       f"ESTIMATED (hidden {ft.hidden_for[i]:.1f} s)" if stt == F.ESTIMATED else "ESTIMATED, +-1 turn")
      info.index = i
      can = dash.compose(states[i], bgr, f.depth_m, idx.history(i), A["buf"], A["pipe"].events.events, info,
                         A["extr"].calibrated)
      # status bar at the bottom of the camera tile
      y = ROW1_Y + ROW1_H - 46
      cv2.rectangle(can, (0, y), (480, ROW1_Y + ROW1_H), (16, 16, 18), -1)
      cv2.rectangle(can, (0, y), (8, ROW1_Y + ROW1_H), STATUS_COL[stt], -1)
      put(can, stt.replace("_", " "), (14, y + 18), 0.55, STATUS_COL[stt], 2)
      tp = t[i] - w0
      qs = "-" if not np.isfinite(yq[i]) else f"{yq[i]:+.1f}"
      sp = "-" if not np.isfinite(spd[i]) else f"{spd[i]:+.0f}"
      put(can, ("policy not started" if tp < 0 else f"policy t {tp:5.1f} s") + f"   turns {qs} qt   speed {sp} deg/s",
          (14, y + 39), 0.47, (230, 230, 230), 1)
      cache[i] = can
    vw.write(cache[i])
  vw.release()
  print(f"[trial-video] {p_out} ({len(ticks)} frames at {a.fps:g} fps, real time)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
