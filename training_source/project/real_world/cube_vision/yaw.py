"""Continuous (unwrapped) cube rotation from successive full orientations.

The six faces carry distinct IDs, so the visual orientation has no cube-symmetry ambiguity:
the increment between two frames is the relative rotation dR = R_k R_{k-1}^T (fixed-frame
increment), and the yaw increment is its rotation vector projected on the yaw axis.  Summing
increments gives the total signed yaw - never a wrapped Euler angle.

Axis conventions
----------------
* palm frame (camera-to-palm calibrated): the project's yaw axis, identical to
  ``HandMonitor.YAW_AXIS_PALM`` = palm -z (positive = +z world yaw in the training mount),
  so vision and proprioceptive cumulative yaw are directly comparable.
* camera frame (not calibrated): the palm normal is unknown.  The axis is either given
  explicitly (camera-frame vector) or estimated as the dominant rotation axis of the motion
  so far (principal direction of the increments).  Its SIGN is then arbitrary and the result
  is labelled "axis estimated from motion"; |yaw| and revolutions are still meaningful.

Gaps
----
While the cube is not tracked nothing is accumulated.  When tracking returns after a gap
(dt > ``max_step_s``) the endpoint difference is *bridged* only if the gap is short
(<= ``max_bridge_s``), the rotation across it is < ``max_bridge_angle_deg`` (shortest path
unambiguous) and below ``max_rate_rad_s`` x dt; otherwise the interval is *excluded* and
reported.  Increments faster than ``max_rate_rad_s`` are rejected as implausible.
Revolutions = |total yaw| / 2 pi (one revolution = four quarter turns).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .geometry import matrix_to_rotvec

YAW_AXIS_PALM = np.array([0.0, 0.0, -1.0])  # == LEAP_Hand_RL_Deploy_Kit/kit/monitor.py HandMonitor.YAW_AXIS_PALM


@dataclass
class Interval:
  t0: float
  t1: float
  kind: str            # "bridged" | "excluded" | "rejected_jump"
  angle_deg: float
  note: str = ""


@dataclass
class YawConfig:
  max_step_s: float = 0.12          # consecutive frames further apart than this are a gap
  max_bridge_s: float = 0.5
  max_bridge_angle_deg: float = 120.0
  max_rate_rad_s: float = 15.0      # matches RobustCubePoseFilter's 18 rad/s order of magnitude
  rate_window_s: float = 0.4


@dataclass
class YawAccumulator:
  axis: np.ndarray | None = None              # fixed axis; None -> estimate from motion
  frame: str = "camera"                        # "palm" | "camera"
  cfg: YawConfig = field(default_factory=YawConfig)

  def __post_init__(self):
    self.reset()

  def reset(self) -> None:
    self._R_prev = None
    self._t_prev = None
    self.increments: list[tuple[float, np.ndarray]] = []  # (t, rotvec increment, fixed frame)
    self.intervals: list[Interval] = []
    self.path_angle_rad = 0.0
    self._lost_since: float | None = None

  # --------------------------------------------------------------- axis
  def current_axis(self) -> tuple[np.ndarray | None, str]:
    if self.axis is not None:
      a = np.asarray(self.axis, np.float64)
      return a / np.linalg.norm(a), ("palm yaw axis (palm -z)" if self.frame == "palm" else "fixed camera-frame axis")
    if len(self.increments) < 3:
      return None, "axis not yet estimated"
    V = np.stack([v for _, v in self.increments])
    M = V.T @ V
    w, U = np.linalg.eigh(M)
    a = U[:, -1]
    if np.sum(V @ a) < 0:  # sign: make the net motion positive (arbitrary but stable)
      a = -a
    return a, "axis estimated from motion (sign arbitrary, not palm-calibrated)"

  # --------------------------------------------------------------- update
  def mark_lost(self, t: float) -> None:
    if self._lost_since is None:
      self._lost_since = t

  def update(self, t: float, R: np.ndarray) -> None:
    R = np.asarray(R, np.float64)
    if self._R_prev is None:
      self._R_prev, self._t_prev, self._lost_since = R, t, None
      return
    dt = t - self._t_prev
    if dt <= 0:
      return
    rv = matrix_to_rotvec(R @ self._R_prev.T)
    ang = float(np.linalg.norm(rv))
    if dt > self.cfg.max_step_s:
      ok = (dt <= self.cfg.max_bridge_s and math.degrees(ang) < self.cfg.max_bridge_angle_deg
            and ang <= self.cfg.max_rate_rad_s * dt)
      self.intervals.append(Interval(self._t_prev, t, "bridged" if ok else "excluded", math.degrees(ang),
                                     f"gap {dt:.2f} s"))
      if ok:
        self._add(t, rv)
    elif ang > self.cfg.max_rate_rad_s * dt:
      self.intervals.append(Interval(self._t_prev, t, "rejected_jump", math.degrees(ang),
                                     f"{ang / dt:.1f} rad/s > {self.cfg.max_rate_rad_s} rad/s"))
    else:
      self._add(t, rv)
    self._R_prev, self._t_prev, self._lost_since = R, t, None

  def _add(self, t: float, rv: np.ndarray) -> None:
    self.increments.append((t, rv))
    self.path_angle_rad += float(np.linalg.norm(rv))

  # --------------------------------------------------------------- outputs
  def yaw_series(self) -> tuple[np.ndarray, np.ndarray]:
    a, _ = self.current_axis()
    if a is None or not self.increments:
      return np.zeros(0), np.zeros(0)
    t = np.array([x[0] for x in self.increments])
    y = np.cumsum([x[1] @ a for x in self.increments])
    return t, y

  @property
  def yaw_rad(self) -> float:
    _, y = self.yaw_series()
    return float(y[-1]) if len(y) else 0.0

  def rate_rad_s(self, t_now: float | None = None) -> float | None:
    t, y = self.yaw_series()
    if len(t) < 3:
      return None
    t_end = t[-1] if t_now is None else t_now
    m = t >= t_end - self.cfg.rate_window_s
    if m.sum() < 3 or (t_now is not None and t_now - t[-1] > self.cfg.max_step_s * 2):
      return None
    A = np.stack([t[m] - t[m][-1], np.ones(m.sum())], axis=1)
    return float(np.linalg.lstsq(A, y[m], rcond=None)[0][0])

  def summary(self) -> dict:
    a, how = self.current_axis()
    y = self.yaw_rad
    excl = [iv for iv in self.intervals if iv.kind != "bridged"]
    return {"yaw_total_rad": y, "yaw_total_deg": math.degrees(y),
            "revolutions": abs(y) / (2 * math.pi), "direction": "+" if y > 0 else ("-" if y < 0 else "0"),
            "quarter_turns": abs(y) / (math.pi / 2), "path_angle_rad": self.path_angle_rad,
            "axis": None if a is None else a.tolist(), "axis_frame": self.frame, "axis_source": how,
            "n_increments": len(self.increments),
            "bridged_intervals": sum(iv.kind == "bridged" for iv in self.intervals),
            "excluded_intervals": len(excl), "excluded_time_s": float(sum(iv.t1 - iv.t0 for iv in excl)),
            "intervals": [iv.__dict__ for iv in self.intervals]}
