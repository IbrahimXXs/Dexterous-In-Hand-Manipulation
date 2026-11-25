#!/usr/bin/env python3
"""Camera-measured cube rotation for every recorded real-hand trial.

Joins the kit's trial record (``LEAP_Hand_RL_Deploy_Kit/logs/trials*.csv`` + each run log) with
the cube-vision dashboard's per-frame poses (``outputs/cube_vision/live_*/poses.jsonl``), which
are on the same host clock.  For each trial it takes the frames of the policy loop
(run log ``started_at`` + ``elapsed_s``) and integrates the cube rotation with occlusion handling
(``cube_vision/gapfill.py``): every gap where fingers hide the cube is bridged by choosing the
whole-turn branch predicted by the speed on both sides, with an uncertainty learned from this
session's own visible motion; gaps that cannot be resolved are flagged, widen the range and are
listed with their times so they can be checked in the recorded video.

  vision_quarter_turns      rotation about the session axis / (pi/2), whole policy loop (signed)
  vision_qt_lo / _hi        range allowing one whole turn either way per ambiguous gap
  vision_qt_until_drop      the same, up to the operator's drop time (if any)
  vision_coverage           share of loop frames with a valid (unambiguous) cube pose
  gaps / gaps_ambiguous     occlusion gaps bridged / of which ambiguous; ambiguous_at_s = their times
  legacy_vision_qt          the previous rule (bridge <= 0.5 s, else drop) for comparison

Axis: without a calibrated camera-to-palm transform the palm normal is unknown in the camera
frame, so ONE dominant rotation axis is estimated from all trials together (first pass) and
used for every trial (second pass): magnitudes are comparable across trials and policies, and a
trial that turns the other way gets the opposite sign.  Which sign is "the trained direction"
is not known from vision alone (it is fixed if --axis-sign-from-trial names a trial the operator
saw turning the trained way).  Tumbling about other axes is not counted as rotation.

    uv run --extra vision python scripts/vision_trials.py \
        --trials LEAP_Hand_RL_Deploy_Kit/logs/trials_20261125.csv \
        --vision outputs/cube_vision/live_20261125_*
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training_source" / "project" / "real_world"))
from cube_vision import gapfill as G  # noqa: E402
from cube_vision.yaw import YawAccumulator  # noqa: E402

KIT_LOGS = ROOT / "LEAP_Hand_RL_Deploy_Kit" / "logs"


def load_frames(dirs: list[Path]) -> tuple[np.ndarray, list[np.ndarray | None]]:
  t, R = [], []
  for d in dirs:
    # a trial folder re-tracked by scripts/trial_video.py (region/range guards, 1-face frames resolved) is preferred
    p = (d / "refined" / "poses.jsonl" if (d / "refined" / "poses.jsonl").exists() else d / "poses.jsonl") if d.is_dir() else d
    if not p.exists():
      continue
    for line in open(p):
      r = json.loads(line)
      ok = r.get("tracking_valid") and not r.get("pose_ambiguous") and r.get("cube_rotation_matrix") is not None
      t.append(float(r["host_time_s"]))
      R.append(np.asarray(r["cube_rotation_matrix"], np.float64) if ok else None)
  order = np.argsort(t)
  return np.asarray(t)[order], [R[i] for i in order]


def run_window(log: Path) -> tuple[float, float]:
  d = np.load(log, allow_pickle=False)
  t0 = datetime.fromisoformat(str(d["started_at"])).timestamp()
  e = d["elapsed_s"]
  return t0 + float(e[0]), t0 + float(e[-1])


def integrate(t: np.ndarray, R: list, t0: float, t1: float, axis: np.ndarray | None) -> tuple[YawAccumulator, int, int]:
  acc = YawAccumulator(axis=axis, frame="camera")
  m = np.where((t >= t0) & (t <= t1))[0]
  valid = 0
  for i in m:
    if R[i] is None:
      acc.mark_lost(t[i])
    else:
      acc.update(t[i], R[i])
      valid += 1
  return acc, len(m), valid


def write_series(trials: Path, r: dict, w: tuple[float, float], gf, observed: str, drop: str) -> None:
  """Per-trial rotation time series (time from policy start) and a plot."""
  base = trials.with_name(f"{trials.stem}_trial{r['trial']}_vision")
  t = gf.series_t - w[0]
  qt = gf.series_y / (math.pi / 2)
  # speed: slope of the cumulative rotation over a centred 1-s window (measured frames only)
  spd = np.full(len(t), np.nan)
  for i in range(len(t)):
    m = np.abs(t - t[i]) <= 0.5
    if m.sum() >= 6 and np.ptp(t[m]) >= 0.5:
      spd[i] = np.polyfit(t[m], np.degrees(gf.series_y[m]), 1)[0]
  with open(base.with_suffix(".csv"), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["t_from_policy_start_s", "host_time_s", "cumulative_quarter_turns", "speed_deg_s"])
    for i in range(len(t)):
      wr.writerow([f"{t[i]:.3f}", f"{gf.series_t[i]:.3f}", f"{qt[i]:.3f}", "" if np.isnan(spd[i]) else f"{spd[i]:.1f}"])
  with open(base.with_name(base.name + "_gaps.csv"), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["start_s", "end_s", "status", "chosen_deg", "shortest_deg", "predicted_deg", "sigma_deg", "p_wrong"])
    for g in gf.gaps:
      wr.writerow([f"{g.t0 - w[0]:.2f}", f"{g.t1 - w[0]:.2f}", g.status, f"{math.degrees(g.chosen):.1f}",
                   f"{math.degrees(g.theta_short):.1f}", "" if g.predicted is None else f"{math.degrees(g.predicted):.1f}",
                   "" if g.sigma is None else f"{math.degrees(g.sigma):.1f}", "" if g.p_wrong is None else f"{g.p_wrong:.3g}"])
  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError:
    return
  fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
  for g in gf.gaps:
    for x in ax:
      x.axvspan(g.t0 - w[0], g.t1 - w[0], color="red" if g.status == "ambiguous" else "0.85", alpha=0.35 if g.status == "ambiguous" else 0.5, lw=0)
  ax[0].plot(t, qt, ".", ms=2, color="C0", label="camera, measured frames")
  ax[0].set_ylabel("cumulative quarter turns")
  if observed not in ("", None):
    ax[0].axhline(float(observed), color="C2", ls="--", label=f"operator count ({float(observed):g})")
  if drop:
    for x in ax:
      x.axvline(float(drop), color="k", ls=":", label="operator: drop")
  ax[0].set_title(f"trial {r['trial']} {r['policy']}: camera {gf.total_rad / (math.pi / 2):+.1f} quarter turns "
                  f"[{gf.lo_rad / (math.pi / 2):+.1f}, {gf.hi_rad / (math.pi / 2):+.1f}]; grey = occluded (bridged), red = ambiguous", fontsize=9)
  ax[0].legend(fontsize=8, loc="upper left")
  ax[1].plot(t, spd, color="C1")
  ax[1].set_ylabel("speed [deg/s] (1-s window)")
  ax[1].set_xlabel("time from policy start [s]")
  fig.tight_layout()
  fig.savefig(base.with_suffix(".png"), dpi=110)
  plt.close(fig)


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--trials", type=Path, required=True)
  ap.add_argument("--vision", nargs="+", required=True, help="dashboard output dir(s) or poses.jsonl file(s); globs ok")
  ap.add_argument("--axis-sign-from-trial", type=int, default=None,
                  help="trial number the operator saw turning the trained (counter-clockwise) way: makes it positive")
  ap.add_argument("--out", type=Path, default=None, help="default: <trials>_vision.csv next to the trials file")
  ap.add_argument("--series", action="store_true",
                  help="also write, per trial, a time series (CSV) and a plot (PNG, needs matplotlib) of the rotation")
  a = ap.parse_args()

  dirs = [Path(p) for g in a.vision for p in (glob.glob(g) or [g])]
  t, R = load_frames(dirs)
  if not len(t):
    sys.exit(f"no vision frames found in {dirs}")
  rows = list(csv.DictReader(open(a.trials)))
  logdir = a.trials.parent
  wins = {}
  for r in rows:
    log = logdir / r["log"]
    if log.exists():
      wins[r["trial"]] = run_window(log)

  # pass 1: one dominant axis from the motion inside all trial windows
  m_all = np.zeros(len(t), bool)
  for w0, w1 in wins.values():
    m_all |= (t >= w0) & (t <= w1)
  t_in, R_in = t[m_all], [R[i] for i in np.where(m_all)[0]]
  axis = G.dominant_axis(t_in, R_in)
  if axis is None:
    sys.exit("too few tracked frames inside the trial windows to estimate a rotation axis")
  # predictability ("accel_sd") is learned per trial, iteratively, from its own visible motion
  cfg = G.GapConfig()
  if a.axis_sign_from_trial is not None and str(a.axis_sign_from_trial) in wins:
    w0, w1 = wins[str(a.axis_sign_from_trial)]
    m = (t >= w0) & (t <= w1)
    if G.integrate_yaw(t[m], [R[i] for i in np.where(m)[0]], axis, cfg).total_rad < 0:
      axis = -axis
    sign_note = f"positive = direction of trial {a.axis_sign_from_trial} (operator: trained direction)"
  else:
    sign_note = "sign arbitrary (same for all trials)"

  out_rows = []
  for r in rows:
    w = wins.get(r["trial"])
    res = {"trial": r["trial"], "policy": r["policy"], "log": r["log"], "observed_quarter_turns": r.get("observed_turns", ""),
           "dropped": r.get("dropped", ""), "drop_time_s": r.get("drop_time_s", "")}
    if w is None:
      res["vision_status"] = "run log missing"
    else:
      m = np.where((t >= w[0]) & (t <= w[1]))[0]
      n, valid = len(m), sum(R[i] is not None for i in m)
      res.update(vision_status="ok" if n else "no vision frames in window", vision_frames=n,
                 vision_coverage=round(valid / n, 3) if n else "")
      if n:
        gf = G.integrate_yaw(t[m], [R[i] for i in m], axis, cfg)
        sm = gf.summary()
        leg, _, _ = integrate(t, R, *w, axis=axis)
        res.update(vision_quarter_turns=round(sm["quarter_turns"], 2), vision_qt_lo=round(sm["quarter_turns_lo"], 2),
                   vision_qt_hi=round(sm["quarter_turns_hi"], 2), gaps=sm["gaps"], gaps_ambiguous=sm["gaps_ambiguous"],
                   gap_time_s=round(sm["gap_time_s"], 2),
                   ambiguous_at_s=" ".join(f"{g.t0 - w[0]:.1f}-{g.t1 - w[0]:.1f}" for g in gf.gaps if g.status == "ambiguous"),
                   legacy_vision_qt=round(leg.yaw_rad / (math.pi / 2), 2), accel_sd=round(gf.accel_sd, 2),
                   accel_sd_windows=gf.accel_sd_windows)
        if a.series:
          write_series(a.trials, r, w, gf, r.get("observed_turns", ""), r.get("drop_time_s") or "")
        dt = r.get("drop_time_s") or ""
        if dt:
          m2 = np.where((t >= w[0]) & (t <= min(w[1], w[0] + float(dt))))[0]
          if len(m2):
            res["vision_qt_until_drop"] = round(G.integrate_yaw(t[m2], [R[i] for i in m2], axis, cfg).total_rad / (math.pi / 2), 2)
    out_rows.append(res)

  out = a.out or a.trials.with_name(a.trials.stem + "_vision.csv")
  keys = ["trial", "policy", "log", "observed_quarter_turns", "dropped", "drop_time_s", "vision_status", "vision_frames",
          "vision_coverage", "vision_quarter_turns", "vision_qt_lo", "vision_qt_hi", "vision_qt_until_drop", "gaps",
          "gaps_ambiguous", "gap_time_s", "ambiguous_at_s", "legacy_vision_qt", "accel_sd", "accel_sd_windows"]
  with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(out_rows)
  (out.with_name(a.trials.stem + "_vision_axis.json")).write_text(json.dumps({"axis": np.asarray(axis).tolist(), "note": sign_note}) + "\n")
  print(f"axis (camera frame) {np.round(axis, 3).tolist()}  [{sign_note}]")
  print("occlusion model: predictability learned per trial from its own visible motion (accel_sd column)")
  print(f"{'trial':>5} {'policy':6} {'observed qt':>11} {'vision qt [range]':>22} {'until drop':>10} {'coverage':>8} {'gaps (ambig.)':>13} {'legacy':>7}")
  for r in out_rows:
    rng_s = f"{r.get('vision_quarter_turns', '-')} [{r.get('vision_qt_lo', '')}, {r.get('vision_qt_hi', '')}]"
    print(f"{r['trial']:>5} {r['policy']:6} {str(r['observed_quarter_turns']):>11} {rng_s:>22} "
          f"{str(r.get('vision_qt_until_drop', '-')):>10} {str(r.get('vision_coverage', r['vision_status'])):>8} "
          f"{str(r.get('gaps', '-')) + ' (' + str(r.get('gaps_ambiguous', '-')) + ')':>13} {str(r.get('legacy_vision_qt', '-')):>7}"
          + (f"   check video at {r['ambiguous_at_s']} s" if r.get("ambiguous_at_s") else ""))
  print(f"-> {out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
