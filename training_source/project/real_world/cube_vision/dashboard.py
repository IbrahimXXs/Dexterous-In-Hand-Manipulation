"""Live / replay digital-twin dashboard (OpenCV window; MuJoCo offscreen for the 3D twin).

Read-only: camera frames, depth, the kit's UDP telemetry or a run log.  It never opens the
serial port, never sends commands, never starts or stops a run.

Layout (1680 x 1000):
  header  mode, source, time, TRACKING/PARTIAL/LOST, depth source, calibration badges
  row 1   camera view (annotated) | depth view | 3D twin (hand + estimates) | camera-frame cube
  row 2   metrics | plots (yaw, yaw rate, position/depth, finger current, joints, tracking error) | events
  footer  replay timeline + controls
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
import math
from pathlib import Path
import time

import numpy as np

from .depth import DepthSource
from .geometry import CubeMarkerModel, matrix_to_rotvec
from .intrinsics import CameraIntrinsics
from .pipeline import FrameState
from .telemetry import FINGER_MOTORS, TelemetryBuffer, command_minus_measured, finger_currents

W, H = 1680, 1000
HEADER = 44
ROW1_Y, ROW1_H = HEADER, 360
ROW2_Y = HEADER + ROW1_H
FOOTER_H = 56
FONT = None  # set on import of cv2

COL = {"bg": (24, 24, 28), "panel": (36, 36, 42), "text": (225, 225, 225), "dim": (150, 150, 155),
       "ok": (90, 200, 90), "warn": (0, 190, 255), "alert": (60, 60, 240), "grid": (60, 60, 66)}
STATE_COL = {"TRACKING": (80, 200, 80), "PARTIAL": (0, 190, 255), "LOST": (60, 60, 230)}
FINGER_BGR = {"thumb": (60, 60, 240), "index": (60, 190, 60), "middle": (255, 130, 60), "ring": (40, 200, 240)}
AXIS_BGR = ((0, 0, 255), (0, 200, 0), (255, 90, 0))


def _cv():
  import cv2

  return cv2


def put(img, text, org, scale=0.5, col=COL["text"], thick=1):
  cv2 = _cv()
  cv2.putText(img, str(text), (int(org[0]), int(org[1])), cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


def fit_into(img, w, h):
  cv2 = _cv()
  ih, iw = img.shape[:2]
  s = min(w / iw, h / ih)
  out = np.full((h, w, 3), COL["panel"], np.uint8)
  r = cv2.resize(img, (max(1, int(iw * s)), max(1, int(ih * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
  y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
  out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
  return out, s, x0, y0


# ----------------------------------------------------------------------------- camera annotation
def annotate_camera(img: np.ndarray, st: FrameState | None, model: CubeMarkerModel, intr: CameraIntrinsics) -> np.ndarray:
  cv2 = _cv()
  out = img.copy()
  if st is None:
    return out
  tr = st.track
  pose = tr.pose
  inl = set(pose.inlier_ids) if pose else set()
  for d in tr.detections:
    col = (80, 220, 80) if d.marker_id in inl else (60, 60, 240)
    q = np.round(d.corners).astype(np.int32)
    cv2.polylines(out, [q], True, col, 2, cv2.LINE_AA)
    cv2.circle(out, tuple(q[0]), 3, (255, 255, 0), -1, cv2.LINE_AA)  # corner 0 (top-left of the marker)
    f = model.faces[d.marker_id]
    put(out, f"{d.marker_id}:{f.letter}({f.name})", q.min(0) + [0, -6], 0.45, col, 1)
  if pose is not None:
    rv, tv = matrix_to_rotvec(pose.R), np.asarray(pose.t, np.float64)
    h = model.half_m
    corners = model.cube_corners()
    pc, _ = cv2.projectPoints(corners, rv, tv, intr.camera_matrix, intr.dist_coeffs)
    pc = pc.reshape(-1, 2)
    for a in range(8):
      for b in range(a + 1, 8):
        if np.sum(np.abs(corners[a] - corners[b]) > 1e-9) == 1:
          cv2.line(out, tuple(np.round(pc[a]).astype(int)), tuple(np.round(pc[b]).astype(int)),
                   (200, 200, 200) if not pose.ambiguous else (0, 190, 255), 1, cv2.LINE_AA)
    ax = np.vstack([np.zeros(3), np.eye(3) * 0.03])
    pa, _ = cv2.projectPoints(ax, rv, tv, intr.camera_matrix, intr.dist_coeffs)
    pa = np.round(pa.reshape(-1, 2)).astype(int)
    for k in range(3):
      cv2.arrowedLine(out, tuple(pa[0]), tuple(pa[k + 1]), AXIS_BGR[k], 2, cv2.LINE_AA, tipLength=0.2)
    cv2.drawMarker(out, tuple(pa[0]), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
  dr = st.depth
  lines = [f"{tr.state.value}  ids {list(tr.visible_ids)}" + (f"  unknown {list(tr.unknown_ids)}" if tr.unknown_ids else "")]
  if pose is not None:
    lines.append(f"reproj {pose.reproj_rms_px:.2f}px  conf {pose.confidence:.2f}" + ("  AMBIGUOUS" if pose.ambiguous else ""))
  if dr is not None and dr.range_m is not None:
    lines.append(f"Z {1000 * dr.z_m:.0f} mm  [{dr.source.value}]")
  for k, s in enumerate(lines):
    put(out, s, (8, 20 + 20 * k), 0.5, (0, 0, 0), 3)
    put(out, s, (8, 20 + 20 * k), 0.5, STATE_COL.get(tr.state.value, COL["text"]) if k == 0 else (255, 255, 255), 1)
  return out


def depth_view(depth_m: np.ndarray | None, st: FrameState | None, w: int, h: int, has_stream: bool) -> np.ndarray:
  cv2 = _cv()
  if depth_m is None:
    p = np.full((h, w, 3), COL["panel"], np.uint8)
    put(p, "NO DEPTH STREAM" if not has_stream else "no depth for this frame", (20, 40), 0.7, COL["warn"], 2)
    if st is not None and st.depth is not None and st.depth.range_m is not None:
      put(p, f"range from {st.depth.source.value}:", (20, 80), 0.55)
      put(p, f"Z {1000 * st.depth.z_m:.1f} mm   |p| {1000 * st.depth.range_m:.1f} mm", (20, 108), 0.55)
      put(p, f"sigma(range) {1000 * st.depth.sigma_range_m:.1f} mm (PnP covariance)", (20, 136), 0.5, COL["dim"])
      put(p, "PNP_METRIC = calibrated geometry + 15.55 mm marker,", (20, 176), 0.45, COL["dim"])
      put(p, "not a measured depth.", (20, 198), 0.45, COL["dim"])
    return p
  d = depth_m.copy()
  valid = np.isfinite(d)
  if st is not None and st.depth is not None and st.depth.z_m:
    lo, hi = st.depth.z_m - 0.08, st.depth.z_m + 0.08
  else:
    lo, hi = (float(np.nanpercentile(d, 2)), float(np.nanpercentile(d, 98))) if valid.any() else (0.2, 1.0)
    if hi - lo < 0.3:  # do not stretch sensor noise over the whole colour map
      mid = 0.5 * (lo + hi)
      lo, hi = mid - 0.15, mid + 0.15
  n = np.clip((np.nan_to_num(d, nan=hi) - lo) / max(hi - lo, 1e-6), 0, 1)
  img = cv2.applyColorMap((255 * (1 - n)).astype(np.uint8), cv2.COLORMAP_TURBO)
  img[~valid] = (90, 0, 90)  # invalid / zero depth
  if st is not None and st.depth is not None:
    for m in st.depth.markers:
      det = next((x for x in st.track.detections if x.marker_id == m.marker_id), None)
      if det is None:
        continue
      c = det.corners.mean(0)
      q = np.round(c + 0.6 * (det.corners - c)).astype(np.int32)
      cv2.polylines(img, [q], True, (255, 255, 255) if m.ok else (0, 0, 255), 1, cv2.LINE_AA)
      if m.median_m is not None:
        put(img, f"{m.marker_id}: {1000 * m.median_m:.0f}mm", q.max(0) + [4, 0], 0.4, (255, 255, 255), 1)
  out, *_ = fit_into(img, w, h)
  dr = st.depth if st is not None else None
  if dr is not None:
    vr = "-" if dr.valid_ratio is None else f"{100 * dr.valid_ratio:.0f}%"
    put(out, f"{dr.source.value}  valid {vr} ({dr.n_valid} px)  conf {dr.confidence:.2f}", (8, 18), 0.45, (255, 255, 255), 1)
    put(out, dr.status[:70], (8, h - 10), 0.4, (255, 255, 255), 1)
  put(out, "magenta = invalid depth", (w - 190, 36), 0.4, (220, 120, 220), 1)
  return out


# ----------------------------------------------------------------------------- plots
def plot(panel: np.ndarray, x0, y0, w, h, title, series, t_range, unit="", y_range=None):
  cv2 = _cv()
  cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), COL["grid"], 1)
  put(panel, title, (x0 + 6, y0 + 15), 0.42, COL["text"])
  ys = [np.asarray(s[1], np.float64) for s in series if len(s[0])]
  ys = [y[np.isfinite(y)] for y in ys]
  ys = [y for y in ys if len(y)]
  if y_range is None:
    if not ys:
      put(panel, "no data", (x0 + w // 2 - 25, y0 + h // 2), 0.45, COL["dim"])
      return
    lo, hi = min(float(y.min()) for y in ys), max(float(y.max()) for y in ys)
    pad = 0.1 * (hi - lo) + 1e-6
    lo, hi = lo - pad, hi + pad
  else:
    lo, hi = y_range
  t0, t1 = t_range
  px0, py0, pw, ph = x0 + 44, y0 + 22, w - 52, h - 30
  for frac in (0.0, 0.5, 1.0):
    yy = int(py0 + ph * (1 - frac))
    cv2.line(panel, (px0, yy), (px0 + pw, yy), COL["grid"], 1)
    put(panel, f"{lo + frac * (hi - lo):.3g}", (x0 + 2, yy + 4), 0.33, COL["dim"])
  put(panel, unit, (x0 + w - 8 - 9 * len(unit), y0 + 15), 0.38, COL["dim"])
  for t, y, col, *lab in series:
    t, y = np.asarray(t, np.float64), np.asarray(y, np.float64)
    if len(t) < 2:
      continue
    u = px0 + (t - t0) / max(t1 - t0, 1e-6) * pw
    v = py0 + (1 - (y - lo) / max(hi - lo, 1e-9)) * ph
    ok = np.isfinite(u) & np.isfinite(v)
    # break the line at gaps
    seg = []
    for k in range(len(u)):
      if ok[k]:
        seg.append((int(u[k]), int(np.clip(v[k], py0 - 2, py0 + ph + 2))))
      elif seg:
        cv2.polylines(panel, [np.array(seg, np.int32)], False, col, 1, cv2.LINE_AA)
        seg = []
    if len(seg) > 1:
      cv2.polylines(panel, [np.array(seg, np.int32)], False, col, 1, cv2.LINE_AA)
  lx = px0 + 4
  for t, y, col, *lab in series:
    if lab and lab[0]:
      put(panel, lab[0], (lx, y0 + h - 4), 0.33, col)
      lx += 9 * len(lab[0]) + 10


# ----------------------------------------------------------------------------- dashboard
@dataclass
class ViewInfo:
  mode: str                 # "LIVE" | "REPLAY"
  source_label: str
  has_depth_stream: bool
  telemetry_label: str
  sync_label: str
  paused: bool = False
  index: int = 0
  n_frames: int = 0
  speed: float = 1.0
  stop_reason: str | None = None


class Dashboard:
  def __init__(self, model: CubeMarkerModel, intr: CameraIntrinsics, with_twin: bool = True):
    from .twin import CubeWidget, TwinRenderer, ensure_gl_backend

    self.model, self.intr = model, intr
    self.widget = CubeWidget(model, 240)
    self.twin = None
    self.twin_error = None
    if with_twin:
      ensure_gl_backend()
      try:
        self.twin = TwinRenderer(model, 480, ROW1_H)
      except Exception as exc:  # noqa: BLE001  - no GL context: dashboard still works without the 3D view
        self.twin_error = f"3D twin unavailable: {exc}"

  def compose(self, st: FrameState | None, bgr: np.ndarray | None, depth_m, history: list[FrameState],
              telemetry: TelemetryBuffer, events: list, info: ViewInfo, extr_ok: bool) -> np.ndarray:
    cv2 = _cv()
    can = np.full((H, W, 3), COL["bg"], np.uint8)
    self._header(can, st, info, extr_ok)
    # camera
    cam = annotate_camera(bgr, st, self.model, self.intr) if bgr is not None else np.full((480, 640, 3), 40, np.uint8)
    can[ROW1_Y:ROW1_Y + ROW1_H, 0:480] = fit_into(cam, 480, ROW1_H)[0]
    can[ROW1_Y:ROW1_Y + ROW1_H, 480:960] = depth_view(depth_m, st, 480, ROW1_H, info.has_depth_stream)
    # twin
    can[ROW1_Y:ROW1_Y + ROW1_H, 960:1440] = self._twin(st, extr_ok)
    # widget + legend
    R = st.track.pose.R if (st is not None and st.track.pose is not None) else None
    can[ROW1_Y:ROW1_Y + 240, 1440:1680] = self.widget.render(R)
    put(can, "vision cube, CAMERA frame", (1446, ROW1_Y + 256), 0.42, COL["text"])
    put(can, "(orientation only; not", (1446, ROW1_Y + 276), 0.4, COL["dim"])
    put(can, " registered to the hand)", (1446, ROW1_Y + 294), 0.4, COL["dim"])
    put(can, "axes: X red, Y green, Z blue", (1446, ROW1_Y + 318), 0.4, COL["dim"])
    for x in (480, 960, 1440):
      cv2.line(can, (x, ROW1_Y), (x, ROW1_Y + ROW1_H), (0, 0, 0), 2)
    self._metrics(can, st, info, telemetry, extr_ok)
    self._plots(can, st, history, telemetry)
    self._events(can, events, st)
    self._footer(can, info, history, events)
    return can

  def _header(self, can, st, info, extr_ok):
    cv2 = _cv()
    cv2.rectangle(can, (0, 0), (W, HEADER), (16, 16, 18), -1)
    mode_col = (60, 60, 230) if info.mode == "LIVE" else (230, 160, 60)
    put(can, info.mode, (10, 30), 0.75, mode_col, 2)
    put(can, info.source_label[:78], (128, 19), 0.43, COL["text"])
    put(can, f"{info.telemetry_label}  |  {info.sync_label}"[:100], (128, 37), 0.38, COL["dim"])
    x = 880
    if st is not None:
      s = st.track.state.value
      cv2.rectangle(can, (x, 8), (x + 120, 36), STATE_COL[s], -1)
      put(can, s, (x + 8, 29), 0.65, (0, 0, 0), 2)
      x += 130
      src = st.depth.source.value if st.depth is not None else DepthSource.DEPTH_UNAVAILABLE.value
      col = {"RGB_D_FUSED": (80, 200, 80), "PNP_METRIC": (0, 190, 255)}.get(src, (60, 60, 230))
      cv2.rectangle(can, (x, 8), (x + 190, 36), col, -1)
      put(can, src, (x + 8, 28), 0.55, (0, 0, 0), 2)
      x += 200
      put(can, f"t {st.t_exp:7.2f} s", (x, 29), 0.55, COL["text"], 1)
    put(can, self.intr.label(), (1390, 18), 0.38, COL["ok"] if self.intr.verified else COL["warn"])
    put(can, "camera->palm: CALIBRATED" if extr_ok else "camera->palm: NOT CALIBRATED", (1390, 37), 0.4,
        COL["ok"] if extr_ok else COL["warn"])

  def _twin(self, st, extr_ok):
    cv2 = _cv()
    if self.twin is None:
      p = np.full((ROW1_H, 480, 3), COL["panel"], np.uint8)
      put(p, (self.twin_error or "3D twin disabled")[:60], (10, 30), 0.45, COL["warn"])
      return p
    q = prop = vis = None
    if st is not None and st.telemetry is not None:
      q = st.telemetry.packet.get("q_server")
      mon = st.telemetry.packet.get("monitor")
      if mon:
        prop = (mon["cube_pos_palm"], mon["cube_quat_palm"])
    if st is not None and st.palm is not None:
      vis = (st.palm["p"], st.palm["R"])
    img = self.twin.render(q, prop, vis)[:, :, ::-1].copy()
    put(img, "hand: measured joints" if q is not None else "hand: NO TELEMETRY (default pose)", (8, 18), 0.45,
        (255, 255, 255) if q is not None else COL["warn"])
    put(img, "orange see-through: proprioceptive estimate" if prop is not None else "no proprioceptive estimate", (8, 38), 0.42,
        (200, 220, 255))
    if extr_ok:
      put(img, "coloured cube: vision (calibrated camera->palm)" if vis is not None else "vision: no valid pose", (8, 58),
          0.42, (200, 255, 200))
    else:
      put(img, "vision cube NOT shown: camera->palm not calibrated", (8, 58), 0.42, COL["warn"])
    return img

  def _metrics(self, can, st, info, telemetry, extr_ok):
    x0, y0, w = 8, ROW2_Y + 14, 520
    rows = []
    if st is None:
      put(can, "waiting for frames", (x0, y0 + 10), 0.55)
      return
    tr, dr, pose = st.track, st.depth, st.track.pose
    letters = " ".join(f"{i}{self.model.faces[i].letter}" for i in tr.visible_ids) or "-"
    rows.append(("tracking", f"{tr.state.value}   markers [{letters}]   inliers {list(pose.inlier_ids) if pose else []}"
                 + (f"  rejected {list(pose.rejected_ids)}" if pose and pose.rejected_ids else "")))
    if pose is not None:
      rows.append(("pose quality", f"reproj {pose.reproj_rms_px:.2f} px   conf {pose.confidence:.2f}   {pose.method}"
                   + ("   AMBIGUOUS (1 face)" if pose.ambiguous else "")))
      p = dr.position_m if (dr is not None and dr.position_m is not None) else pose.t
      rows.append(("cube XYZ cam", "  ".join(f"{1000 * v:+7.1f}" for v in p) + " mm"
                   + (f"   std {' '.join(f'{1000 * s:.1f}' for s in pose.pos_std_m)} mm" if pose.pos_std_m is not None else "")))
      rv = matrix_to_rotvec(pose.R)
      ang = np.degrees(np.linalg.norm(rv))
      axis = rv / max(np.linalg.norm(rv), 1e-9)
      rows.append(("orientation cam", f"{ang:6.1f} deg about [{axis[0]:+.2f} {axis[1]:+.2f} {axis[2]:+.2f}]"))
    else:
      rows += [("pose quality", "-"), ("cube XYZ cam", "-"), ("orientation cam", "-")]
    if extr_ok and st.palm is not None:
      rows.append(("cube XYZ palm", "  ".join(f"{1000 * v:+7.1f}" for v in st.palm["p"]) + f" mm   |d| "
                   f"{1000 * np.linalg.norm(st.palm['p']):.1f} mm"))
    else:
      rows.append(("cube XYZ palm", "NOT CALIBRATED - palm-frame displacement not shown"))
    if dr is not None and dr.range_m is not None:
      vr = "-" if dr.valid_ratio is None else f"{100 * dr.valid_ratio:.0f}%"
      rows.append(("depth", f"Z {1000 * dr.z_m:.1f} mm  range {1000 * dr.range_m:.1f} +- {1000 * dr.sigma_range_m:.1f} mm"
                   f"  [{dr.source.value}]"))
      rows.append(("depth samples", f"valid {vr}  ({dr.n_valid}/{dr.n_sampled} px)  conf {dr.confidence:.2f}"))
    else:
      rows += [("depth", "DEPTH_UNAVAILABLE"), ("depth samples", "-")]
    rate = "-" if st.yaw_rate is None else f"{math.degrees(st.yaw_rate):+.1f} deg/s"
    rows.append(("rotation", f"{math.degrees(st.yaw_rad):+.1f} deg  = {st.revolutions:.3f} rev "
                 f"({'+' if st.yaw_rad >= 0 else '-'})  {abs(st.yaw_rad) / (math.pi / 2):.2f} quarter turns"))
    rows.append(("rotation rate", f"{rate}   axis: {st.yaw_axis_source}"))
    rows.append(("elapsed", f"{st.t_exp:.2f} s   frame {tr.frame}" + (f"   video {st.video_time_s:.2f} s" if st.video_time_s is not None else "")))
    if st.telemetry is not None:
      pk = st.telemetry.packet
      fc = finger_currents(pk)
      mode = "SHADOW (torque off)" if pk.get("shadow") else "POWERED"
      if pk.get("port") == "SIMULATED":
        mode = "SIMULATED bus"
      rows.append(("telemetry", f"step {pk.get('step')}  t_run {pk.get('t', 0):.2f} s  age {1000 * (st.telemetry_age_s or 0):.0f} ms  {mode}"))
      rows.append(("finger current", "  ".join(f"{f[0].upper()} {v:.0f}" for f, v in fc.items()) + " mA (measured, not force)"))
      mon = pk.get("monitor")
      if mon:
        rows.append(("proprio est.", f"turns {float(mon['est_yaw_rad']) / (2 * math.pi):+.3f} (integrated estimate)"))
    else:
      rows.append(("telemetry", "none associated with this frame"))
    c = st.comparison
    if c is not None:
      sp = "-" if c["rotation_speed_difference_deg_s"] is None else f"{c['rotation_speed_difference_deg_s']:+.1f}"
      rows.append(("vision-proprio", f"pos {c['position_disagreement_mm']:.1f} mm  ori {c['orientation_disagreement_deg_mod_symmetry']:.1f} deg (mod sym.)"))
      rows.append(("", f"cum. yaw diff {c['cumulative_yaw_difference_deg']:+.1f} deg   speed diff {sp} deg/s"))
    elif not extr_ok:
      rows.append(("vision-proprio", "needs camera->palm calibration (not compared)"))
    lines = []
    for a, b in rows:  # wrap to the panel width (~54 characters at this font size)
      b = str(b)
      first = True
      while b:
        cut = 54 if len(b) > 54 else len(b)
        if cut < len(b) and " " in b[:cut]:
          cut = b[:cut].rindex(" ")
        lines.append((a if first else "", b[:cut]))
        b, first = b[cut:].lstrip(), False
    for k, (a, b) in enumerate(lines[:30]):
      put(can, a, (x0, y0 + 18 * k + 10), 0.41, COL["dim"])
      put(can, b, (x0 + 118, y0 + 18 * k + 10), 0.41, COL["text"])

  def _plots(self, can, st, history, telemetry):
    x0, y0 = 540, ROW2_Y + 6
    pw, ph = 445, 152
    if st is None:
      return
    t_now = st.t_exp
    tr = (t_now - 20.0, t_now + 0.5)
    t = np.array([s.t_exp for s in history])
    yaw = np.array([math.degrees(s.yaw_rad) for s in history])
    rate = np.array([np.nan if s.yaw_rate is None else math.degrees(s.yaw_rate) for s in history])
    pos = np.array([(s.depth.position_m if (s.depth is not None and s.depth.position_m is not None) else
                     (s.track.pose.t if s.track.pose is not None else [np.nan] * 3)) for s in history]) * 1000 \
      if history else np.zeros((0, 3))
    valid = np.array([s.track.tracking_valid for s in history], bool)
    if len(pos):
      pos[~valid] = np.nan
    # telemetry over the same window (host time = frame host time - t_exp offset)
    host_of = st.track.host_time_s - st.t_exp
    tel = telemetry.window(host_of + tr[0], host_of + tr[1])
    tt = np.array([s.host_time_s - host_of for s in tel])
    p_yaw = np.array([float(s.packet["monitor"]["est_yaw_rad"]) if s.packet.get("monitor") else np.nan for s in tel])
    series_yaw = [(t, yaw, (80, 220, 80), "vision")]
    if len(tt) and np.isfinite(p_yaw).any():
      ref_v = yaw[0] if len(yaw) else 0.0
      pv = np.degrees(p_yaw - p_yaw[np.isfinite(p_yaw)][0]) + ref_v
      series_yaw.append((tt, pv, (90, 160, 255), "proprio est. (aligned at window start)"))
    plot(can, x0, y0, pw, ph, "cumulative yaw", series_yaw, tr, "deg")
    plot(can, x0 + pw + 5, y0, pw, ph, "yaw rate (vision)", [(t, rate, (80, 220, 80), "")], tr, "deg/s")
    ser = [(t, pos[:, k] if len(pos) else [], AXIS_BGR[k], "xyz"[k]) for k in range(2)]
    plot(can, x0, y0 + ph + 4, pw, ph, "cube X, Y (camera frame)", ser, tr, "mm")
    zser = [(t, pos[:, 2] if len(pos) else [], AXIS_BGR[2], "Z final")]
    zd = np.array([np.nan if (s.depth is None or s.depth.depth_range_m is None) else 1000 * s.depth.depth_range_m for s in history])
    if np.isfinite(zd).any():
      zser.append((t, zd, (200, 200, 200), "|p| depth stream"))
    plot(can, x0 + pw + 5, y0 + ph + 4, pw, ph, "cube depth Z / range", zser, tr, "mm")
    if len(tel):
      cur = np.array([[finger_currents(s.packet)[f] for f in FINGER_MOTORS] for s in tel])
      plot(can, x0, y0 + 2 * (ph + 4), pw, ph, "finger current (sum |I|, measured)",
           [(tt, cur[:, k], FINGER_BGR[f], f) for k, f in enumerate(FINGER_MOTORS)], tr, "mA")
      err = np.array([np.abs(command_minus_measured(s.packet)) for s in tel])
      ser = []
      for k, (f, pre) in enumerate((("index", 0), ("middle", 4), ("ring", 8), ("thumb", 12))):
        ser.append((tt, err[:, pre:pre + 4].max(axis=1), FINGER_BGR[f], f))
      plot(can, x0 + pw + 5, y0 + 2 * (ph + 4), pw, ph, "tracking error max |cmd - q| per finger", ser, tr, "rad")
    else:
      plot(can, x0, y0 + 2 * (ph + 4), pw, ph, "finger current (no telemetry)", [], tr, "mA")
      plot(can, x0 + pw + 5, y0 + 2 * (ph + 4), pw, ph, "tracking error (no telemetry)", [], tr, "rad")

  def _events(self, can, events, st):
    cv2 = _cv()
    x0, y0 = 1444, ROW2_Y + 10
    put(can, "events", (x0, y0 + 6), 0.5, COL["text"])
    t_now = st.t_exp if st is not None else 1e18
    shown = [e for e in events if e.t <= t_now + 1e-6][-24:][::-1]
    y = y0 + 26
    for e in shown:
      col = {"warn": COL["warn"], "alert": (90, 90, 255)}.get(e.severity, COL["dim"])
      put(can, f"{e.t:6.1f}s {e.kind}", (x0, y), 0.38, col)
      msg = e.message
      for part in (msg[:34], msg[34:68]):
        if part:
          y += 12
          put(can, part, (x0 + 6, y), 0.32, COL["text"])
      y += 16
      if y > H - FOOTER_H - 24:
        break

  def _footer(self, can, info, history, events):
    cv2 = _cv()
    y = H - FOOTER_H
    cv2.rectangle(can, (0, y), (W, H), (16, 16, 18), -1)
    if info.mode == "REPLAY" and info.n_frames:
      x0, x1 = 20, W - 420
      cv2.line(can, (x0, y + 22), (x1, y + 22), COL["grid"], 6)
      px = int(x0 + (x1 - x0) * info.index / max(info.n_frames - 1, 1))
      cv2.line(can, (x0, y + 22), (px, y + 22), (230, 160, 60), 6)
      cv2.circle(can, (px, y + 22), 8, (255, 255, 255), -1)
      put(can, f"frame {info.index}/{info.n_frames - 1}  x{info.speed:g}  {'PAUSED' if info.paused else 'PLAYING'}",
          (x1 + 20, y + 26), 0.5, COL["text"])
      put(can, "space play/pause   , . step   [ ] speed   drag 'frame' slider   s screenshot   q quit", (20, y + 48), 0.42,
          COL["dim"])
    else:
      put(can, "LIVE  -  read-only: no commands are sent to the hand    s screenshot   q quit", (20, y + 30), 0.5, COL["dim"])
    if info.stop_reason:
      put(can, f"run log stop_reason: {info.stop_reason}", (W - 400, y + 48), 0.42, COL["warn"])


# ----------------------------------------------------------------------------- replay model
class ReplayIndex:
  """Precomputed FrameStates for a recording, so scrubbing shows states consistent with the
  sequential yaw accumulation."""

  def __init__(self, states: list[FrameState]):
    self.states = states
    self.t = [s.t_exp for s in states]

  def history(self, i: int, span_s: float = 25.0) -> list[FrameState]:
    t = self.t[i]
    j = bisect.bisect_left(self.t, t - span_s)
    return self.states[j:i + 1]
