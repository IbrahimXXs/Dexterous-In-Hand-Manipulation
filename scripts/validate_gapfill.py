#!/usr/bin/env python3
"""Validate occlusion handling (cube_vision/gapfill.py) against the current gap rule.

A. Synthetic ground truth: 60-s sequences, 30 fps, tilted spin axis, smoothly varying speed with
   occasional reversals, wobble (swing) and 0.5 deg pose noise, occlusion gaps drawn like the real
   ones (median ~0.3 s, up to 4 s, ~25% of the time hidden).
B. Real data: the hand-held recording evidence/videos/rgb.mp4, tracked with the measured layout.
   Windows of continuous tracking get an artificial hidden interval of L seconds in the middle;
   the true rotation across it is known from the hidden frames themselves.

Both methods get the same axis.  "legacy" = YawAccumulator (bridge <= 0.5 s and < 120 deg, else
exclude); "gapfill" = speed-predicted branch selection with an ambiguity flag.

    uv run --extra vision python scripts/validate_gapfill.py [--poses outputs/cube_vision/rgb_mp4_candidate/poses.jsonl]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training_source" / "project" / "real_world"))
from cube_vision.gapfill import GapConfig, _rot_angle, dominant_axis, integrate_yaw, twist_angle  # noqa: E402
from cube_vision.yaw import YawAccumulator  # noqa: E402

QT = math.pi / 2


def rotm(axis, ang):
  a = np.asarray(axis, float) / np.linalg.norm(axis)
  K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
  return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def legacy(t, Rs, axis):
  acc = YawAccumulator(axis=axis, frame="camera")
  for ti, R in zip(t, Rs):
    if R is None:
      acc.mark_lost(ti)
    else:
      acc.update(ti, R)
  return acc.yaw_rad


def synthetic(rng, fps=30.0, T=60.0):
  n = int(T * fps)
  t = np.arange(n) / fps
  axis = rng.normal(size=3)
  axis /= np.linalg.norm(axis)
  base = rng.uniform(0.3, 2.0) * rng.choice([1, -1])
  rate = np.empty(n)
  r = base
  for i in range(n):  # OU speed around the base, occasional reversal
    r += (base - r) * 0.5 / fps + rng.normal(0, 1.2 / math.sqrt(fps))
    if rng.random() < 0.02 / fps:
      base = -base
    rate[i] = r
  yaw = np.cumsum(rate) / fps
  perp = np.cross(axis, [1, 0, 0] if abs(axis[0]) < 0.9 else [0, 1, 0])
  perp /= np.linalg.norm(perp)
  R0 = rotm(rng.normal(size=3), rng.uniform(0, math.pi))
  Rs = []
  for i in range(n):
    swing = rotm(perp, math.radians(8) * math.sin(2 * math.pi * 0.7 * t[i]))
    noise = rotm(rng.normal(size=3), math.radians(abs(rng.normal(0, 0.5))))
    Rs.append(noise @ swing @ rotm(axis, yaw[i]) @ R0)
  # occlusion gaps
  vis = np.ones(n, bool)
  i = int(rng.uniform(0.5, 2) * fps)
  while i < n:
    L = int(min(4.0, rng.lognormal(math.log(0.3), 1.0)) * fps)
    vis[i:i + L] = False
    i += L + int(rng.exponential(1.2) * fps) + 3
  vis[:3] = vis[-3:] = True
  # truth: rotation about the axis between the first and last frame (sum of the yaw, the swing returns)
  truth = sum(twist_angle(Rs[k + 1] @ Rs[k].T, axis) for k in range(n - 1))
  return t, [R if v else None for R, v in zip(Rs, vis)], axis, truth, 1 - vis.mean()


def part_a(nseq: int):
  rng = np.random.default_rng(0)
  err_new, err_old, amb, cover, hidden = [], [], [], [], []
  for _ in range(nseq):
    t, Rs, axis, truth, h = synthetic(rng)
    r = integrate_yaw(t, Rs, axis)
    err_new.append((r.total_rad - truth) / QT)
    err_old.append((legacy(t, Rs, axis) - truth) / QT)
    amb.append(sum(g.status == "ambiguous" for g in r.gaps))
    cover.append(r.lo_rad - 1e-6 <= truth <= r.hi_rad + 1e-6)
    hidden.append(h)
  en, eo = np.abs(err_new), np.abs(err_old)
  print(f"\nA. synthetic ground truth: {nseq} x 60 s, cube hidden {100 * np.mean(hidden):.0f}% of the time")
  print(f"   error in quarter turns per 60 s     median   p90    max   |err| >= 1 qt")
  print(f"   legacy (bridge <=0.5 s, else drop)  {np.median(eo):6.2f} {np.percentile(eo, 90):6.2f} {eo.max():6.2f}   {np.mean(eo >= 1) * 100:5.1f}%")
  print(f"   gapfill (speed-predicted)           {np.median(en):6.2f} {np.percentile(en, 90):6.2f} {en.max():6.2f}   {np.mean(en >= 1) * 100:5.1f}%")
  print(f"   gapfill: ambiguous gaps per sequence {np.mean(amb):.2f}; truth inside [lo, hi] in {np.mean(cover) * 100:.0f}% of sequences")
  return {"n": nseq, "hidden_frac": float(np.mean(hidden)),
          "legacy_abs_err_qt": {"median": float(np.median(eo)), "p90": float(np.percentile(eo, 90)), "max": float(eo.max()), "ge1": float(np.mean(eo >= 1))},
          "gapfill_abs_err_qt": {"median": float(np.median(en)), "p90": float(np.percentile(en, 90)), "max": float(en.max()), "ge1": float(np.mean(en >= 1))},
          "ambiguous_per_seq": float(np.mean(amb)), "bounds_cover_truth": float(np.mean(cover))}


def part_b(poses: Path):
  rows = [json.loads(l) for l in open(poses)]
  t = np.array([r.get("video_time_s") if r.get("video_time_s") is not None else r["host_time_s"] for r in rows], float)
  Rs = [np.asarray(r["cube_rotation_matrix"]) if (r.get("tracking_valid") and not r.get("pose_ambiguous")
        and r.get("cube_rotation_matrix") is not None) else None for r in rows]
  cfg = GapConfig()
  axis = dominant_axis(t, Rs, cfg)
  # contiguous runs of valid frames (no step > max_step, no implausible jump)
  runs, cur = [], []
  for i, R in enumerate(Rs):
    ok = R is not None and (not cur or (t[i] - t[cur[-1]] <= cfg.max_step_s
                                         and _rot_angle(R @ Rs[cur[-1]].T) <= cfg.max_rate_rad_s * (t[i] - t[cur[-1]])))
    if ok:
      cur.append(i)
    else:
      if len(cur) > 1:
        runs.append(cur)
      cur = [i] if R is not None else []
  if len(cur) > 1:
    runs.append(cur)
  print(f"\nB. real hand-held recording ({poses}): {len(runs)} continuous runs, longest {max(t[r[-1]] - t[r[0]] for r in runs):.1f} s")
  print(f"   hidden L   windows   legacy wrong(|err|>45deg)  gapfill wrong   gapfill ambiguous   median |err| legacy / gapfill (deg)")
  out = {}
  for L in (0.5, 1.0, 1.5, 2.0, 3.0):
    eo, en, amb = [], [], []
    for run in runs:
      tr = t[run]
      if tr[-1] - tr[0] < L + 2.0:
        continue
      # slide windows of (1 s, L hidden, 1 s) through the run, non-overlapping
      s0 = tr[0]
      while s0 + L + 2.0 <= tr[-1]:
        idx = [i for i in run if s0 <= t[i] <= s0 + L + 2.0]
        Rw = [None if s0 + 1.0 < t[i] < s0 + 1.0 + L else Rs[i] for i in idx]
        tw = t[idx]
        truth = sum(twist_angle(Rs[idx[k + 1]] @ Rs[idx[k]].T, axis) for k in range(len(idx) - 1))
        r = integrate_yaw(tw, Rw, axis, cfg)
        en.append(abs(math.degrees(r.total_rad - truth)))
        eo.append(abs(math.degrees(legacy(tw, Rw, axis) - truth)))
        amb.append(any(g.status == "ambiguous" for g in r.gaps))
        s0 += L + 2.0
    if not en:
      continue
    en, eo, amb = np.array(en), np.array(eo), np.array(amb)
    print(f"   {L:4.1f} s   {len(en):7d}   {np.mean(eo > 45) * 100:21.1f}%   {np.mean(en > 45) * 100:11.1f}%   {np.mean(amb) * 100:15.1f}%"
          f"        {np.median(eo):6.1f} / {np.median(en):5.1f}")
    out[f"{L}"] = {"windows": int(len(en)), "legacy_wrong_frac": float(np.mean(eo > 45)), "gapfill_wrong_frac": float(np.mean(en > 45)),
                   "gapfill_ambiguous_frac": float(np.mean(amb)), "gapfill_wrong_when_resolved": float(np.mean((en > 45) & ~amb)),
                   "legacy_median_abs_err_deg": float(np.median(eo)), "gapfill_median_abs_err_deg": float(np.median(en))}
  # whole recording, natural gaps
  full_new = integrate_yaw(t, Rs, axis, cfg).summary()
  full_old = legacy(t, Rs, axis)
  print(f"   whole recording (natural gaps): legacy {full_old / QT:+.1f} qt, gapfill {full_new['quarter_turns']:+.1f} qt "
        f"[{full_new['quarter_turns_lo']:+.1f}, {full_new['quarter_turns_hi']:+.1f}], gaps {full_new['gaps']} "
        f"({full_new['gaps_resolved']} resolved, {full_new['gaps_ambiguous']} ambiguous)")
  out["whole_recording"] = {"legacy_qt": full_old / QT, **full_new}
  return out


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--poses", type=Path, default=ROOT / "outputs/cube_vision/rgb_mp4_candidate/poses.jsonl")
  ap.add_argument("--nseq", type=int, default=200)
  ap.add_argument("--out", type=Path, default=ROOT / "training_source/project/real_world/cube_vision_gapfill_validation.json")
  a = ap.parse_args()
  res = {"synthetic": part_a(a.nseq)}
  if a.poses.exists():
    res["real_rgb_mp4"] = part_b(a.poses)
  a.out.write_text(json.dumps(res, indent=1) + "\n")
  print(f"\n-> {a.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
