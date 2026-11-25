"""Occlusion-aware cumulative rotation (offline, both sides of every gap known).

While fingers hide the cube no pose is measured.  Because every face has its own marker ID,
the orientation after re-acquisition is known exactly, so the rotation about the spin axis
during a gap is known *up to whole turns*:

    rotation_in_gap = theta_short + 2*pi*k ,   theta_short in (-pi, pi]   (k integer, unknown)

The integer k is resolved from the rotation speed measured just before and just after the gap
(constant-acceleration interpolation), with an uncertainty that grows with the gap length.  A gap
is *resolved* when the probability that the true rotation belongs to another branch is below
``max_wrong_branch_prob``; otherwise it is *ambiguous*: the most likely branch is used for the
total and the neighbouring branches widen the reported [lo, hi] range.  Nothing is excluded
silently, unlike a fixed "bridge only short gaps" rule, which drops the rotation of long gaps.

Rotation about the axis is the exact *twist* of the swing-twist decomposition, so tilting or
wobbling of the cube during a gap does not leak into the count.

Validated against ground truth on synthetic sequences and on the real hand-held recording with
artificially hidden intervals (``scripts/validate_gapfill.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

TWO_PI = 2.0 * math.pi


@dataclass
class GapConfig:
  max_step_s: float = 0.12            # consecutive valid frames further apart than this = a gap
  max_rate_rad_s: float = 15.0        # implausible single-step rotation (bad pose) -> frame dropped
  rate_window_s: float = 0.8          # window at each segment end used to estimate the speed
  rate_window_long_s: float = 1.5     # window on segments joined across already-resolved gaps
  max_iterations: int = 4
  min_rate_span_s: float = 0.2        # at least this much segment time for a speed estimate
  rate_sigma_floor: float = 0.08      # rad/s, floor on the speed uncertainty
  accel_sd: float | str = "auto"      # rad/s^2, unmodelled change of speed inside a gap; "auto" = learned
                                      # from this session's own visible motion (calibrate_accel_sd)
  accel_sd_default: float = 2.5       # used when there is too little visible motion to calibrate
  accel_sd_floor: float = 0.3
  max_wrong_branch_prob: float = 0.01
  legacy_bridge_s: float = 0.5        # fallback when no speed is available on either side
  legacy_bridge_deg: float = 120.0


@dataclass
class Gap:
  t0: float
  t1: float
  theta_short: float          # shortest-path rotation about the axis across the gap
  predicted: float | None     # from the speeds on both sides
  sigma: float | None
  chosen: float               # rotation used for the total
  status: str                 # resolved | ambiguous | shortest_no_speed
  p_wrong: float | None = None

  @property
  def dt(self) -> float:
    return self.t1 - self.t0


@dataclass
class GapfillResult:
  total_rad: float
  lo_rad: float
  hi_rad: float
  axis: list
  segments: int
  gaps: list = field(default_factory=list)
  dropped_frames: int = 0
  tracked_time_s: float = 0.0
  gap_time_s: float = 0.0

  def summary(self) -> dict:
    g = self.gaps
    return {"total_rad": self.total_rad, "quarter_turns": self.total_rad / (math.pi / 2),
            "quarter_turns_lo": self.lo_rad / (math.pi / 2), "quarter_turns_hi": self.hi_rad / (math.pi / 2),
            "revolutions": self.total_rad / TWO_PI, "axis": self.axis, "segments": self.segments,
            "gaps": len(g), "gaps_resolved": sum(x.status == "resolved" for x in g),
            "gaps_ambiguous": sum(x.status == "ambiguous" for x in g),
            "gaps_shortest_no_speed": sum(x.status == "shortest_no_speed" for x in g),
            "gap_time_s": self.gap_time_s, "tracked_time_s": self.tracked_time_s,
            "accel_sd": getattr(self, "accel_sd", None), "accel_sd_calibration_windows": getattr(self, "accel_sd_windows", None),
            "dropped_frames": self.dropped_frames}


def _quat(R: np.ndarray) -> np.ndarray:
  m = R
  w = 0.5 * math.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2]))
  x = math.copysign(0.5 * math.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])), m[2, 1] - m[1, 2])
  y = math.copysign(0.5 * math.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])), m[0, 2] - m[2, 0])
  z = math.copysign(0.5 * math.sqrt(max(0.0, 1 - m[0, 0] - m[1, 1] + m[2, 2])), m[1, 0] - m[0, 1])
  q = np.array([w, x, y, z])
  return q / np.linalg.norm(q)


def twist_angle(R_rel: np.ndarray, axis: np.ndarray) -> float:
  """Rotation of R_rel about ``axis`` (swing-twist decomposition), in (-pi, pi]."""
  q = _quat(np.asarray(R_rel, np.float64))
  a = np.asarray(axis, np.float64) / np.linalg.norm(axis)
  ang = 2.0 * math.atan2(float(q[1:] @ a), float(q[0]))
  return (ang + math.pi) % TWO_PI - math.pi if ang > math.pi or ang <= -math.pi else ang


def _rot_angle(R_rel: np.ndarray) -> float:
  return math.acos(max(-1.0, min(1.0, (np.trace(R_rel) - 1.0) / 2.0)))


def dominant_axis(t: np.ndarray, Rs: list, cfg: GapConfig | None = None) -> np.ndarray | None:
  """Principal axis of the frame-to-frame rotation increments of contiguous valid frames."""
  cfg = cfg or GapConfig()
  V = []
  prev = None
  for ti, R in zip(t, Rs):
    if R is None:
      prev = None
      continue
    if prev is not None and 0 < ti - prev[0] <= cfg.max_step_s:
      Rr = R @ prev[1].T
      ang = _rot_angle(Rr)
      if ang <= cfg.max_rate_rad_s * (ti - prev[0]) and ang > 1e-9:
        w = np.array([Rr[2, 1] - Rr[1, 2], Rr[0, 2] - Rr[2, 0], Rr[1, 0] - Rr[0, 1]])
        V.append(w / max(np.linalg.norm(w), 1e-12) * ang)
    prev = (ti, R)
  if len(V) < 3:
    return None
  V = np.asarray(V)
  a = np.linalg.eigh(V.T @ V)[1][:, -1]
  return a if np.sum(V @ a) >= 0 else -a


def _segment_series(t, Rs, axis, cfg):
  """Contiguous segments: list of (times, cumulative twist, first R, last R); frames whose step
  rotation is implausibly fast are dropped (bad pose), not treated as rotation."""
  segs, dropped = [], 0
  cur_t, cur_y, first_R, last = [], [], None, None
  for ti, R in zip(t, Rs):
    if R is None:
      continue
    if last is not None and ti - last[0] <= cfg.max_step_s:
      Rr = R @ last[1].T
      if _rot_angle(Rr) > cfg.max_rate_rad_s * max(ti - last[0], 1e-3):
        dropped += 1
        continue
      cur_t.append(ti)
      cur_y.append(cur_y[-1] + twist_angle(Rr, axis))
      last = (ti, R)
      continue
    if cur_t:
      segs.append((np.array(cur_t), np.array(cur_y), first_R, last[1]))
    cur_t, cur_y, first_R, last = [ti], [0.0], R, (ti, R)
  if cur_t:
    segs.append((np.array(cur_t), np.array(cur_y), first_R, last[1]))
  return segs, dropped


def _speed(tt: np.ndarray, yy: np.ndarray, at_end: bool, cfg: GapConfig, window: float | None = None):
  """Speed (rad/s) and its standard error at the start or the end of a segment."""
  window = cfg.rate_window_s if window is None else window
  if tt[-1] - tt[0] < cfg.min_rate_span_s or len(tt) < 4:
    return None, None
  m = tt >= tt[-1] - window if at_end else tt <= tt[0] + window
  x, y = tt[m] - tt[m].mean(), yy[m]
  if len(x) < 4 or np.ptp(x) < cfg.min_rate_span_s * 0.5:
    return None, None
  A = np.c_[x, np.ones_like(x)]
  coef, *_ = np.linalg.lstsq(A, y, rcond=None)
  res = y - A @ coef
  se = math.sqrt(max(float(res @ res) / max(len(x) - 2, 1), 0.0) / max(float(x @ x), 1e-9))
  return float(coef[0]), max(se, cfg.rate_sigma_floor)


def _p_wrong(d: float, sigma: float) -> float:
  """P(|true - chosen| > pi) with true ~ N(predicted, sigma) and chosen - predicted = d."""
  s = max(sigma, 1e-9) * math.sqrt(2.0)
  return 0.5 * math.erfc((math.pi + d) / s) + 0.5 * math.erfc((math.pi - d) / s)


def _predict(tt, yy, nt, ny, cfg, window: float | None = None):
  """Speeds (and standard errors) on both sides of a gap; None if neither side has a speed."""
  ra, sa = _speed(tt, yy, True, cfg, window)
  rb, sb = _speed(nt, ny, False, cfg, window)
  if ra is None and rb is None:
    return None
  if ra is None:
    ra, sa = rb, sb
  if rb is None:
    rb, sb = ra, sa
  return ra, sa, rb, sb


def _sigma(dt, ra, sa, rb, sb, accel_sd):
  return math.sqrt((0.5 * dt) ** 2 * (sa ** 2 + sb ** 2) + (accel_sd ** 2) * dt ** 3 / 12.0
                   + ((rb - ra) * dt / 4.0) ** 2)


def calibrate_accel_sd(segs, cfg: GapConfig, window: float | None = None) -> tuple[float, int]:
  """How predictable is the rotation here?  Hide intervals of 0.3-3 s inside this session's own
  continuous segments, predict across them exactly as for a real gap, and choose accel_sd so that
  the prediction errors match the model's variance (trimmed mean of the excess squared error)."""
  ratios = []
  w = cfg.rate_window_s if window is None else window
  for seg in segs:
    tt, yy = seg[0], seg[1]
    for L in (0.3, 0.6, 1.0, 1.5, 2.0, 3.0):
      s0 = tt[0]
      while s0 + L + 2 * w <= tt[-1]:
        a = (tt >= s0) & (tt <= s0 + w)
        b = (tt >= s0 + w + L) & (tt <= s0 + 2 * w + L)
        if a.sum() >= 4 and b.sum() >= 4:
          pr = _predict(tt[a], yy[a], tt[b], yy[b], cfg, w)
          if pr is not None:
            ra, sa, rb, sb = pr
            dt = tt[b][0] - tt[a][-1]
            err = (yy[b][0] - yy[a][-1]) - 0.5 * (ra + rb) * dt
            base = _sigma(dt, ra, sa, rb, sb, 0.0) ** 2
            ratios.append(max(err * err - base, 0.0) / (dt ** 3 / 12.0))
        s0 += L + 2 * w
  if len(ratios) < 8:
    return cfg.accel_sd_default, len(ratios)
  r = np.sort(np.asarray(ratios))[: max(1, int(0.95 * len(ratios)))]  # trim the top 5% (bad poses)
  return max(cfg.accel_sd_floor, math.sqrt(float(r.mean()))), len(ratios)


def _solve(left, right, th, dt, accel_sd, cfg, window):
  """Branch choice for one gap from the series on its left and right."""
  pr = _predict(left[0], left[1], right[0], right[1], cfg, window)
  if pr is None:
    ok = dt <= cfg.legacy_bridge_s and abs(math.degrees(th)) < cfg.legacy_bridge_deg
    return th, ("shortest_no_speed" if ok else "ambiguous"), None, None, None
  ra, sa, rb, sb = pr
  pred = 0.5 * (ra + rb) * dt
  sigma = _sigma(dt, ra, sa, rb, sb, accel_sd)
  chosen = th + TWO_PI * round((pred - th) / TWO_PI)
  pw = _p_wrong(chosen - pred, sigma)
  return chosen, ("resolved" if pw < cfg.max_wrong_branch_prob else "ambiguous"), pred, sigma, pw


def _groups(segs, status):
  """Join consecutive segments across gaps that are settled (resolved or a short shortest-path
  bridge) into continuous series; returns [(first_seg, last_seg, T, Y)] with Y continuous."""
  out, i = [], 0
  while i < len(segs):
    j, T, Y, off = i, [segs[i][0]], [segs[i][1]], segs[i][1][-1]
    while j < len(segs) - 1 and status[j][1] in ("resolved", "shortest_no_speed"):
      off += status[j][0]
      T.append(segs[j + 1][0])
      Y.append(segs[j + 1][1] + off)
      off += segs[j + 1][1][-1]
      j += 1
    out.append((i, j, np.concatenate(T), np.concatenate(Y)))
    i = j + 1
  return out


def integrate_yaw(t, Rs, axis=None, cfg: GapConfig | None = None) -> GapfillResult:
  """Cumulative rotation about ``axis`` (estimated if None) over valid frames (R None = no pose).

  Iterative: gaps are first solved from the speed on the bare visible segments; the segments are
  then joined across the gaps that were resolved, and the remaining gaps are re-solved from
  speeds over the longer joined series (and the predictability is re-learned from them), until
  nothing changes.  Short visible bursts between occlusions therefore still yield good speeds."""
  cfg = cfg or GapConfig()
  t = np.asarray(t, np.float64)
  if axis is None:
    axis = dominant_axis(t, Rs, cfg)
    if axis is None:
      return GapfillResult(0.0, 0.0, 0.0, [], 0)
  axis = np.asarray(axis, np.float64) / np.linalg.norm(axis)
  segs, dropped = _segment_series(t, Rs, axis, cfg)
  auto = cfg.accel_sd == "auto"
  if auto:
    accel_sd, n_cal = calibrate_accel_sd(segs, cfg)
  else:
    accel_sd, n_cal = float(cfg.accel_sd), 0
  ng = len(segs) - 1
  th = [twist_angle(segs[k + 1][2] @ segs[k][3].T, axis) for k in range(ng)]
  dts = [segs[k + 1][0][0] - segs[k][0][-1] for k in range(ng)]
  status = [None] * ng  # (chosen, status, pred, sigma, p_wrong)
  for k in range(ng):
    status[k] = _solve(segs[k][:2], segs[k + 1][:2], th[k], dts[k], accel_sd, cfg, cfg.rate_window_s)
  for _ in range(cfg.max_iterations):
    if all(st[1] != "ambiguous" for st in status):
      break
    groups = _groups(segs, status)
    if auto:
      a2, n2 = calibrate_accel_sd([(g[2], g[3]) for g in groups], cfg, cfg.rate_window_long_s)
      if n2 >= 8 and n2 > n_cal:
        accel_sd, n_cal = a2, n2
    changed = False
    for gi in range(len(groups) - 1):
      k = groups[gi][1]  # the unsettled gap between this group and the next
      new = _solve((groups[gi][2], groups[gi][3]), (groups[gi + 1][2], groups[gi + 1][3]), th[k], dts[k],
                   accel_sd, cfg, cfg.rate_window_long_s)
      if new[1] != "ambiguous" and status[k][1] == "ambiguous":
        changed = True
      if new[1] != "ambiguous" or new[2] is not None:
        status[k] = new
    if not changed:
      break
  total = lo = hi = float(sum(sg[1][-1] for sg in segs))
  gaps = []
  for k in range(ng):
    chosen, st, pred, sigma, pw = status[k]
    total += chosen
    lo += chosen - (TWO_PI if st == "ambiguous" else 0.0)
    hi += chosen + (TWO_PI if st == "ambiguous" else 0.0)
    gaps.append(Gap(segs[k][0][-1], segs[k + 1][0][0], th[k], pred, sigma, chosen, st, pw))
  tracked = float(sum(sg[0][-1] - sg[0][0] for sg in segs))
  res = GapfillResult(total, lo, hi, axis.tolist(), len(segs), gaps, dropped, tracked,
                      float(sum(g.dt for g in gaps)))
  res.accel_sd, res.accel_sd_windows = accel_sd, n_cal
  # cumulative rotation at every measured frame (gap rotations as chosen), for plots / exports
  st, sy, cum = [], [], 0.0
  for k, sg in enumerate(segs):
    st.append(sg[0])
    sy.append(cum + sg[1])
    cum += sg[1][-1] + (status[k][0] if k < ng else 0.0)
  res.series_t = np.concatenate(st) if st else np.zeros(0)
  res.series_y = np.concatenate(sy) if sy else np.zeros(0)
  return res
