"""Per-frame vision pipeline: tracker -> depth fusion -> jump filter -> yaw -> palm frame ->
telemetry association -> vision-vs-proprioception comparison -> events.  Pure computation;
used identically by the live dashboard, replay and headless export."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np

from .depth import DepthConfig, DepthResult, DepthSource, estimate_depth
from .events import EventConfig, EventLog
from .extrinsics import ExtrinsicsStatus
from .geometry import CubeMarkerModel, cube_symmetry_rotations, matrix_from_quat_wxyz, quat_wxyz_from_matrix, \
  rotation_angle_deg
from .intrinsics import CameraIntrinsics
from .telemetry import SERVER_JOINT_NAMES, TelemetryBuffer, TelemetrySample, joint_currents_server, read_hand_env
from .tracker import CubeTracker, TrackerConfig, TrackResult, TrackState
from .yaw import YAW_AXIS_PALM, YawAccumulator, YawConfig

_SYM = cube_symmetry_rotations()


@dataclass
class FrameState:
  track: TrackResult
  depth: DepthResult | None
  video_time_s: float | None
  t_exp: float                       # experiment time (host time - first frame)
  intrinsics_status: str
  layout_status: str
  filter_status: str
  yaw_rad: float
  yaw_rate: float | None
  revolutions: float
  yaw_axis_source: str
  palm: dict | None
  extrinsics_message: str
  telemetry: TelemetrySample | None
  telemetry_age_s: float | None
  comparison: dict | None


@dataclass
class PipelineConfig:
  telemetry_max_age_s: float = 0.15     # camera frame <-> 20 Hz telemetry association window
  telemetry_offset_s: float = 0.0       # added to frame host time before lookup (sync of recordings)
  history_s: float = 30.0
  camera_yaw_axis: tuple[float, float, float] | None = None   # camera-frame axis when uncalibrated
  # optional guards (off by default): markers are only detected inside roi_xyxy (pixels; e.g. the hand
  # region, so a second cube elsewhere in view cannot be tracked), and a pose whose camera Z is not in
  # (0.05 m, max_range_m] is dropped (behind the camera / far away = not the cube in the hand)
  roi_xyxy: tuple[int, int, int, int] | None = None
  max_range_m: float | None = None


class VisionPipeline:
  def __init__(self, model: CubeMarkerModel, intr: CameraIntrinsics, extr: ExtrinsicsStatus, *,
               tracker_cfg: TrackerConfig | None = None, depth_cfg: DepthConfig | None = None,
               event_cfg: EventConfig | None = None, cfg: PipelineConfig | None = None,
               telemetry: TelemetryBuffer | None = None):
    try:
      from ..tracked_cube_filter import RobustCubePoseFilter
    except ImportError:  # pragma: no cover
      from tracked_cube_filter import RobustCubePoseFilter
    self.model, self.intr, self.extr = model, intr, extr
    self.tracker = CubeTracker(model, intr, tracker_cfg)
    self.depth_cfg = depth_cfg or DepthConfig()
    self.cfg = cfg or PipelineConfig()
    hand = read_hand_env()
    ecfg = event_cfg or EventConfig()
    if ecfg.high_current_ma is None and "CURRENT_LIMIT_MA" in hand:
      try:
        ecfg.high_current_ma = 0.9 * float(hand["CURRENT_LIMIT_MA"])
      except ValueError:
        pass
    self.hand_env = hand
    self.events = EventLog(ecfg)
    self.filter = RobustCubePoseFilter(position_time_constant_s=0.0, orientation_time_constant_s=0.0)
    if extr.calibrated:
      self.yaw = YawAccumulator(YAW_AXIS_PALM.copy(), "palm", YawConfig())
    else:
      ax = None if self.cfg.camera_yaw_axis is None else np.asarray(self.cfg.camera_yaw_axis, np.float64)
      self.yaw = YawAccumulator(ax, "camera", YawConfig())
    self.telemetry = telemetry or TelemetryBuffer()
    self.t_first: float | None = None
    self.hist = deque()
    self.prev_valid: tuple[float, np.ndarray, np.ndarray] | None = None
    self.proprio_hist = deque(maxlen=400)
    self.cmp_ref: tuple[float, float] | None = None  # (vision yaw, proprio yaw) at first comparable frame
    self.layout_status = (f"{model.layout_path.name} + print layout {model.print_layout.status}"
                          if model.print_layout_path is not None else f"{model.layout_path.name} ({model.print_layout.status})")
    self.state_counts: dict[str, int] = {}
    self.depth_counts: dict[str, int] = {}
    self.reproj: list[float] = []
    self.layout_conflict_frames = 0
    self.multi_face_frames = 0
    self.n = 0

  # ------------------------------------------------------------------ main
  def process(self, frame, video_time_s: float | None = None) -> FrameState:
    if self.t_first is None:
      self.t_first = frame.host_time_s
    t_exp = frame.host_time_s - self.t_first
    img = frame.bgr
    if self.cfg.roi_xyxy is not None:
      x0, y0, x1, y1 = self.cfg.roi_xyxy
      img = np.zeros_like(frame.bgr)
      img[y0:y1, x0:x1] = frame.bgr[y0:y1, x0:x1]
    tr = self.tracker.process(img, frame.host_time_s, frame.camera_time_s)
    if self.cfg.max_range_m is not None and tr.pose is not None and not 0.05 < float(tr.pose.t[2]) <= self.cfg.max_range_m:
      self.events.add(t_exp, "region", f"pose dropped: camera Z {float(tr.pose.t[2]):.3g} m outside (0.05, "
                      f"{self.cfg.max_range_m:g}] m", "warn", tr.frame)
      if self.tracker.prev is not None and self.tracker.prev[0] == frame.host_time_s:
        self.tracker.prev = None
      tr.pose, tr.state = None, TrackState.LOST
    ev = self.events
    ev.visibility(t_exp, tr.frame, tr.visible_ids)
    ev.tracking_state(t_exp, tr.frame, tr.state.value)
    if tr.pose is not None and tr.pose.layout_conflict_ids:
      self.layout_conflict_frames += 1
      ev.layout(t_exp, tr.frame, tr.pose.layout_conflict_ids, len(tr.pose.inlier_ids))
    dr = estimate_depth(tr, frame.depth_m, self.intr, self.model, self.depth_cfg)
    if frame.depth_m is not None and tr.pose is not None:
      ev.depth(t_exp, tr.frame, dr.source == DepthSource.RGB_D_FUSED, dr.status)
    filt = "-"
    palm = None
    if tr.tracking_valid:
      pos = dr.position_m if dr.position_m is not None else tr.pose.t
      try:
        self.filter.update(position_palm_m=pos, quaternion_palm_wxyz=quat_wxyz_from_matrix(tr.pose.R),
                           timestamp_s=frame.host_time_s, source_frame=tr.frame, visible_marker_ids=tr.visible_ids)
        filt = "accepted"
        accept = True
      except ValueError as exc:  # implausible jump (RobustCubePoseFilter rules) or time reversal
        filt = f"rejected: {exc}"
        accept = False
        ev.add(t_exp, "filter", f"pose rejected by jump filter: {exc}", "warn", tr.frame)
      if accept:
        if self.prev_valid is not None:
          dt = frame.host_time_s - self.prev_valid[0]
          ev.motion(t_exp, tr.frame, float(np.linalg.norm(pos - self.prev_valid[1])), dt,
                    rotation_angle_deg(self.prev_valid[2], tr.pose.R))
        self.prev_valid = (frame.host_time_s, pos.copy(), tr.pose.R.copy())
        if self.extr.calibrated:
          T = self.extr.transform
          p_palm, R_palm = T.point(pos), T.rotation(tr.pose.R)
          palm = {"p": p_palm, "R": R_palm, "q": quat_wxyz_from_matrix(R_palm), "file": T.path}
          self.yaw.update(frame.host_time_s, R_palm)
        else:
          self.yaw.update(frame.host_time_s, tr.pose.R)
        rng = float(np.linalg.norm(pos))
        lo, hi = ev.cfg.region_range_m
        inside, why = lo <= rng <= hi, f"range {1000 * rng:.0f} mm outside [{1000 * lo:.0f}, {1000 * hi:.0f}] mm"
        if inside and palm is not None and ev.cfg.region_palm_radius_m:
          d = float(np.linalg.norm(palm["p"]))
          inside, why = d <= ev.cfg.region_palm_radius_m, f"{1000 * d:.0f} mm from the palm centre"
        ev.region(t_exp, tr.frame, inside, why)
    else:
      self.yaw.mark_lost(frame.host_time_s)
    # telemetry association (causal, explicit staleness)
    tel, age = self.telemetry.nearest(frame.host_time_s + self.cfg.telemetry_offset_s, self.cfg.telemetry_max_age_s)
    if tel is not None:
      thr = ev.cfg.high_current_ma
      ev.current(t_exp, tr.frame, joint_currents_server(tel.packet), SERVER_JOINT_NAMES, thr)
    cmp = self._compare(palm, tel, frame.host_time_s)
    ysum_axis = self.yaw.current_axis()[1]
    y = self.yaw.yaw_rad
    st = FrameState(tr, dr, video_time_s, t_exp, self.intr.label(), self.layout_status, filt, y,
                    self.yaw.rate_rad_s(frame.host_time_s), abs(y) / (2 * math.pi), ysum_axis, palm,
                    self.extr.message, tel, age, cmp)
    self._stats(st)
    self.hist.append(st)
    while self.hist and self.hist[0].t_exp < t_exp - self.cfg.history_s:
      self.hist.popleft()
    self.n += 1
    return st

  def _compare(self, palm, tel, t_host) -> dict | None:
    """Vision (external visual measurement) vs the kit's proprioceptive estimate.  Only with a
    calibrated camera-to-palm transform; the proprioceptive orientation is symmetry-ambiguous,
    so its disagreement is taken modulo the 24 cube symmetries."""
    if tel is None or not tel.packet.get("monitor"):
      return None
    mon = tel.packet["monitor"]
    p_yaw = float(mon["est_yaw_rad"])
    self.proprio_hist.append((t_host, p_yaw))
    if palm is None:
      return None
    Rp = matrix_from_quat_wxyz(np.asarray(mon["cube_quat_palm"], np.float64))
    ang = min(rotation_angle_deg(palm["R"], Rp @ S) for S in _SYM)
    v_yaw = self.yaw.yaw_rad
    if self.cmp_ref is None:
      self.cmp_ref = (v_yaw, p_yaw)
    rate_v = self.yaw.rate_rad_s(t_host)
    h = [x for x in self.proprio_hist if x[0] >= t_host - 0.4]
    rate_p = None
    if len(h) >= 3:
      tt = np.array([x[0] for x in h])
      A = np.stack([tt - tt[-1], np.ones(len(tt))], axis=1)
      rate_p = float(np.linalg.lstsq(A, np.array([x[1] for x in h]), rcond=None)[0][0])
    return {"position_disagreement_mm": float(1000 * np.linalg.norm(palm["p"] - np.asarray(mon["cube_pos_palm"]))),
            "orientation_disagreement_deg_mod_symmetry": float(ang),
            "cumulative_yaw_difference_deg": float(np.degrees((v_yaw - self.cmp_ref[0]) - (p_yaw - self.cmp_ref[1]))),
            "rotation_speed_difference_deg_s": None if (rate_v is None or rate_p is None) else float(np.degrees(rate_v - rate_p)),
            "vision_yaw_since_ref_deg": float(np.degrees(v_yaw - self.cmp_ref[0])),
            "proprio_yaw_since_ref_deg": float(np.degrees(p_yaw - self.cmp_ref[1]))}

  def _stats(self, st: FrameState) -> None:
    self.state_counts[st.track.state.value] = self.state_counts.get(st.track.state.value, 0) + 1
    if st.depth is not None:
      self.depth_counts[st.depth.source.value] = self.depth_counts.get(st.depth.source.value, 0) + 1
    if st.track.pose is not None:
      self.reproj.append(st.track.pose.reproj_rms_px)
    if len(st.track.visible_ids) >= 2:
      self.multi_face_frames += 1

  def summary(self, mode: str, source: str) -> dict:
    n = max(self.n, 1)
    cmp = [s.comparison for s in self.hist if s.comparison]
    out = {"mode": mode, "source": source, "frames": self.n,
           "duration_s": (self.hist[-1].t_exp if self.hist else 0.0),
           "layout": self.layout_status, "intrinsics": self.intr.label(), "intrinsics_file": self.intr.source_path,
           "extrinsics": self.extr.message,
           "state_fraction": {k: v / n for k, v in sorted(self.state_counts.items())},
           "depth_source_fraction": {k: v / n for k, v in sorted(self.depth_counts.items())},
           "median_reprojection_px": float(np.median(self.reproj)) if self.reproj else None,
           "yaw": self.yaw.summary(),
           "multi_face_frames": self.multi_face_frames, "layout_conflict_frames": self.layout_conflict_frames,
           "event_counts": _count(e.kind for e in self.events.events),
           "events": [e.__dict__ for e in self.events.events]}
    if cmp:
      out["comparison"] = {k: {"median": float(np.nanmedian([c[k] for c in cmp if c[k] is not None])),
                               "max_abs": float(np.nanmax([abs(c[k]) for c in cmp if c[k] is not None]))}
                           for k in ("position_disagreement_mm", "orientation_disagreement_deg_mod_symmetry",
                                     "cumulative_yaw_difference_deg") if any(c[k] is not None for c in cmp)}
    return out


def _count(it) -> dict:
  d: dict = {}
  for k in it:
    d[k] = d.get(k, 0) + 1
  return d
