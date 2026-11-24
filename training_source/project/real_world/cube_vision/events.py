"""Experiment event timeline.  Each event states what the data show - nothing is inferred
beyond it (no contact or safety claims from vision)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Event:
  t: float              # experiment time [s] (host time - first frame)
  kind: str
  message: str
  severity: str = "info"  # info | warn | alert
  frame: int | None = None


@dataclass
class EventConfig:
  debounce_frames: int = 3
  jump_translation_m: float = 0.015
  jump_speed_m_s: float = 0.6
  large_rotation_step_deg: float = 30.0
  large_rotation_rate_rad_s: float = 8.0
  high_current_ma: float | None = None   # default: 0.9 x CURRENT_LIMIT_MA from hand.env
  telemetry_timeout_s: float = 1.0
  region_range_m: tuple[float, float] = (0.10, 1.20)   # camera-frame range of the cube centre
  region_palm_radius_m: float | None = 0.12            # palm-frame distance, only when calibrated


@dataclass
class EventLog:
  cfg: EventConfig = field(default_factory=EventConfig)
  events: list[Event] = field(default_factory=list)

  def __post_init__(self):
    self._vis_candidate = None
    self._vis_count = 0
    self._vis_current = None
    self._state_candidate = None
    self._state_count = 0
    self._state_current = None
    self._high_cur: set[int] = set()
    self._telemetry_alive = None
    self._in_region = None
    self._depth_ok = None
    self._layout_warned: set[int] = set()

  def add(self, t, kind, msg, severity="info", frame=None) -> Event:
    e = Event(float(t), kind, msg, severity, frame)
    self.events.append(e)
    return e

  def visibility(self, t, frame, ids: tuple[int, ...]) -> None:
    if ids == self._vis_candidate:
      self._vis_count += 1
    else:
      self._vis_candidate, self._vis_count = ids, 1
    if self._vis_count == self.cfg.debounce_frames and ids != self._vis_current:
      prev = self._vis_current
      self._vis_current = ids
      if prev is not None:
        self.add(t, "markers", f"visible markers {list(prev)} -> {list(ids)}", frame=frame)

  def tracking_state(self, t, frame, state: str) -> None:
    if state == self._state_candidate:
      self._state_count += 1
    else:
      self._state_candidate, self._state_count = state, 1
    if self._state_count == self.cfg.debounce_frames and state != self._state_current:
      prev = self._state_current
      self._state_current = state
      if state == "LOST" and prev is not None:
        self.add(t, "tracking", "tracking LOST (no reliable marker)", "warn", frame)
      elif prev == "LOST":
        self.add(t, "tracking", f"tracking recovered ({state})", "info", frame)

  def motion(self, t, frame, dp_m: float, dt: float, drot_deg: float) -> None:
    if dt <= 0:
      return
    if dp_m > self.cfg.jump_translation_m or dp_m / dt > self.cfg.jump_speed_m_s:
      self.add(t, "translation", f"sudden cube translation {1000 * dp_m:.1f} mm in {1000 * dt:.0f} ms", "warn", frame)
    rate = drot_deg / 57.29578 / dt
    if drot_deg > self.cfg.large_rotation_step_deg or rate > self.cfg.large_rotation_rate_rad_s:
      self.add(t, "rotation", f"large orientation change {drot_deg:.0f} deg in {1000 * dt:.0f} ms", "warn", frame)

  def layout(self, t, frame, rejected: tuple[int, ...], n_faces: int) -> None:
    for i in rejected:
      if i not in self._layout_warned:
        self._layout_warned.add(i)
        self.add(t, "layout", f"marker {i} fits well alone but contradicts the other visible face(s) under the "
                              "loaded layout - the layout file does not match this cube", "alert", frame)

  def current(self, t, frame, cur_server, names, threshold) -> None:
    if threshold is None:
      return
    for j, c in enumerate(cur_server):
      hi = abs(c) >= threshold
      if hi and j not in self._high_cur:
        self._high_cur.add(j)
        self.add(t, "actuator", f"high motor current {names[j]} {abs(c):.0f} mA (>= {threshold:.0f} mA; measured "
                                f"current, not a force)", "warn", frame)
      elif not hi and j in self._high_cur and abs(c) < 0.8 * threshold:
        self._high_cur.discard(j)

  def telemetry(self, t, alive: bool, detail: str = "") -> None:
    if alive != self._telemetry_alive:
      if self._telemetry_alive is not None or alive:
        self.add(t, "telemetry", ("robot telemetry received" if alive else
                                  f"robot telemetry stopped {detail}".strip()), "info" if alive else "warn")
      self._telemetry_alive = alive

  def region(self, t, frame, inside: bool, why: str) -> None:
    if inside != self._in_region:
      if self._in_region is not None:
        self.add(t, "region", ("cube back inside the tracking region" if inside else f"cube left the tracking region: {why}"),
                 "info" if inside else "warn", frame)
      self._in_region = inside

  def depth(self, t, frame, ok: bool | None, status: str) -> None:
    if ok is None:
      return
    if ok != self._depth_ok:
      if self._depth_ok is not None:
        self.add(t, "depth", ("RGB-D fusion resumed" if ok else f"RGB-D depth not fused: {status}"), "info", frame)
      self._depth_ok = ok

  def guard(self, t, stop_reason: str) -> None:
    sev = "alert" if ("guard" in stop_reason.lower() or "error" in stop_reason.lower()) else "info"
    self.add(t, "run_end", f"run log stop_reason: {stop_reason}", sev)
