"""Per-frame cube state through occlusions (offline, uses frames on both sides of every gap).

The live tracker reports a pose only when it is sure: one visible face has the two-fold IPPE ambiguity
(two candidate orientations), and a hidden cube has none.  After a trial, every frame can be filled:

1. ``gapfill.integrate_yaw`` on the measured frames gives the rotation axis, the cumulative rotation at
   each measured frame and, for every gap, the number of turns made while hidden (speed-predicted branch,
   with a probability that the branch is wrong).
2. Single-face frames inside a *resolved* gap: the orientation predicted for that instant (interpolated
   across the gap, below) picks one of the two IPPE candidates, if exactly one is close (<= 25 deg).
   Those frames become measurements, which splits long gaps into short ones, so 1. is repeated
   (up to 3 times) until nothing changes.
3. Every remaining frame between two measurements is ESTIMATED: rotation about the axis follows a cubic
   Hermite curve through both ends with the measured speeds at both ends (so the gap's chosen number of
   turns is distributed smoothly); the off-axis part (wobble) is interpolated geometrically; position is
   interpolated linearly.  Frames inside a gap whose branch is uncertain are flagged as such.

Nothing is extrapolated beyond the last view (after a drop the state is NOT_VISIBLE)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import gapfill as G
from .geometry import matrix_to_rotvec, rotation_angle_deg, rotvec_to_matrix

MEASURED = "MEASURED"                  # pose from >= 2 faces, or 1 face resolved by the tracker itself
RESOLVED = "RESOLVED_1FACE"            # 1 face, IPPE ambiguity resolved here from both sides of the gap
ESTIMATED = "ESTIMATED"                # hidden / unusable frame inside a resolved gap
UNCERTAIN = "ESTIMATED_UNCERTAIN"      # hidden inside a gap whose number of turns is not certain (+-1 turn)
NOT_VISIBLE = "NOT_VISIBLE"            # before the first or after the last view: no estimate


@dataclass
class FillConfig:
  gap: G.GapConfig = field(default_factory=G.GapConfig)
  resolve_max_deg: float = 25.0        # a 1-face candidate is accepted if it is this close to the prediction
  resolve_max_deg_uncertain: float = 12.0
  resolve_min_separation_deg: float = 60.0
  resolve_iterations: int = 3
  slope_window_s: float = 0.6          # speed at a gap end: fit over this much of the visible side


@dataclass
class FilledTrack:
  t: np.ndarray                        # frame host times
  status: list                         # per frame, one of the constants above
  yaw: np.ndarray                      # cumulative rotation about the axis [rad] (NaN where NOT_VISIBLE)
  R: list                              # per frame 3x3 orientation (camera frame) or None
  p: list                              # per frame cube centre (camera frame, m) or None
  gap_of: np.ndarray                   # per frame index into result.gaps, -1 if not in a gap
  hidden_for: np.ndarray               # per frame: length of the gap it is in [s] (0 if measured)
  result: G.GapfillResult
  axis: np.ndarray
  resolved_frames: int = 0
  choice: dict = field(default_factory=dict)  # frame -> (R, p) chosen 1-face candidate


def _rot(axis: np.ndarray, ang: float) -> np.ndarray:
  return rotvec_to_matrix(np.asarray(axis, np.float64) * ang)


def _end_slope(tt: np.ndarray, yy: np.ndarray, at_end: bool, w: float) -> float:
  m = tt >= tt[-1] - w if at_end else tt <= tt[0] + w
  if m.sum() < 3 or np.ptp(tt[m]) < 0.1:
    return float("nan")
  return float(np.polyfit(tt[m], yy[m], 1)[0])


def _interp_track(t, Rs, P, res: G.GapfillResult, axis, cfg: FillConfig):
  """Per-frame yaw / R / p / status from a gapfill result over the measured frames (Rs[i] not None)."""
  n = len(t)
  st_t, st_y = res.series_t, res.series_y
  idx_of = {float(x): k for k, x in enumerate(st_t)}
  meas = np.array([i for i in range(n) if Rs[i] is not None and float(t[i]) in idx_of], int)
  yaw = np.full(n, np.nan)
  R_out, p_out = [None] * n, [None] * n
  status = [NOT_VISIBLE] * n
  gap_of = np.full(n, -1)
  hidden = np.zeros(n)
  for i in meas:
    yaw[i] = st_y[idx_of[float(t[i])]]
    R_out[i], p_out[i], status[i] = Rs[i], P[i], MEASURED
  if len(meas) < 2:
    return yaw, R_out, p_out, status, gap_of, hidden
  gap_by_t0 = {round(g.t0, 6): (k, g) for k, g in enumerate(res.gaps)}
  # speeds at both ends of every interval between consecutive measured frames
  for a, b in zip(meas[:-1], meas[1:]):
    if b - a < 2:
      continue
    ta, tb = float(t[a]), float(t[b])
    ya, yb = yaw[a], yaw[b]
    kg, g = gap_by_t0.get(round(ta, 6), (-1, None))
    left = (st_t <= ta) & (st_t >= ta - cfg.slope_window_s)
    right = (st_t >= tb) & (st_t <= tb + cfg.slope_window_s)
    ma = _end_slope(st_t[left], st_y[left], True, cfg.slope_window_s) if left.sum() >= 3 else float("nan")
    mb = _end_slope(st_t[right], st_y[right], False, cfg.slope_window_s) if right.sum() >= 3 else float("nan")
    chord = (yb - ya) / max(tb - ta, 1e-6)
    ma = chord if not np.isfinite(ma) else ma
    mb = chord if not np.isfinite(mb) else mb
    T = tb - ta
    resid = matrix_to_rotvec(Rs[b] @ (_rot(axis, yb - ya) @ Rs[a]).T)  # off-axis part not explained by the twist
    for i in range(a + 1, b):
      s = (float(t[i]) - ta) / T
      h00, h10, h01, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
      yaw[i] = h00 * ya + h10 * T * ma + h01 * yb + h11 * T * mb
      R_out[i] = rotvec_to_matrix(resid * s) @ _rot(axis, yaw[i] - ya) @ Rs[a]
      if P[a] is not None and P[b] is not None:
        p_out[i] = (1 - s) * np.asarray(P[a]) + s * np.asarray(P[b])
      status[i] = UNCERTAIN if (g is not None and g.status == "ambiguous") else ESTIMATED
      gap_of[i], hidden[i] = kg, T
  return yaw, R_out, p_out, status, gap_of, hidden


def fill_track(t, Rs, P, candidates=None, axis=None, cfg: FillConfig | None = None) -> FilledTrack:
  """t: frame times; Rs: measured orientation or None; P: position or None;
  candidates: per frame None or a list of (R, p) for an ambiguous single-face pose."""
  cfg = cfg or FillConfig()
  t = np.asarray(t, np.float64)
  n = len(t)
  Rm = list(Rs)
  Pm = list(P)
  candidates = candidates or [None] * n
  if axis is None:
    axis = G.dominant_axis(t, Rm, cfg.gap)
  if axis is None:
    res = G.GapfillResult(0.0, 0.0, 0.0, [], 0)
    res.series_t, res.series_y = np.zeros(0), np.zeros(0)
    return FilledTrack(t, [NOT_VISIBLE] * n, np.full(n, np.nan), [None] * n, [None] * n, np.full(n, -1), np.zeros(n),
                       res, np.zeros(3))
  axis = np.asarray(axis, np.float64) / np.linalg.norm(axis)
  chosen: dict = {}
  for _ in range(cfg.resolve_iterations + 1):
    res = G.integrate_yaw(t, Rm, axis, cfg.gap)
    yaw, R_est, p_est, status, gap_of, hidden = _interp_track(t, Rm, Pm, res, axis, cfg)
    new = 0
    for i in range(n):
      c = candidates[i]
      if Rm[i] is not None or not c or status[i] not in (ESTIMATED, UNCERTAIN) or R_est[i] is None:
        continue
      d = sorted((rotation_angle_deg(R_est[i], Rc), k) for k, (Rc, _) in enumerate(c))
      # inside a gap whose number of turns is uncertain the prediction is weaker: a candidate must match it
      # closely AND the other one must be far off (a wrong-branch path would not land on either by chance)
      tol = cfg.resolve_max_deg if status[i] == ESTIMATED else cfg.resolve_max_deg_uncertain
      if d[0][0] <= tol and (status[i] == ESTIMATED or len(d) < 2 or d[1][0] >= cfg.resolve_min_separation_deg):
        Rc, pc = c[d[0][1]]
        Rm[i], Pm[i] = Rc, pc
        chosen[i] = (Rc, pc)
        new += 1
    if not new:
      break
  res = G.integrate_yaw(t, Rm, axis, cfg.gap)
  yaw, R_est, p_est, status, gap_of, hidden = _interp_track(t, Rm, Pm, res, axis, cfg)
  for i in chosen:
    if status[i] == MEASURED:
      status[i] = RESOLVED
  return FilledTrack(t, status, yaw, R_est, p_est, gap_of, hidden, res, axis, len(chosen), chosen)


def speed_series(t: np.ndarray, yaw: np.ndarray, window_s: float = 1.0) -> np.ndarray:
  """Rotation speed [rad/s]: slope of a line fit over a centred window (NaN where too little data)."""
  out = np.full(len(t), np.nan)
  ok = np.isfinite(yaw)
  for i in range(len(t)):
    if not ok[i]:
      continue
    m = ok & (np.abs(t - t[i]) <= window_s / 2)
    if m.sum() >= 3 and np.ptp(t[m]) >= 0.3 * window_s:
      out[i] = np.polyfit(t[m], yaw[m], 1)[0]
  return out


def status_counts(status: list) -> dict:
  n = max(len(status), 1)
  return {s: round(sum(x == s for x in status) / n, 3) for s in (MEASURED, RESOLVED, ESTIMATED, UNCERTAIN, NOT_VISIBLE)}


__all__ = ["FillConfig", "FilledTrack", "fill_track", "speed_series", "status_counts",
           "MEASURED", "RESOLVED", "ESTIMATED", "UNCERTAIN", "NOT_VISIBLE"]
