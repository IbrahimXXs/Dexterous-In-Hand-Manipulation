#!/usr/bin/env python3
"""Termination diagnostic (dev seed 6104): are pose-deviation "failures" real grasp losses?

The task's ``cube_pose_deviation`` termination fires when the cube's tilt from its
EPISODE-INITIAL attitude exceeds 0.8 rad (or drift > 8 cm).  A cube that starts propped on
an edge (palm-frame tilt ~0.7 rad, the "bad" grasps) and settles flat onto a face is
therefore terminated although it is seated more securely.  Here the same policies are rolled
out with that termination switched off (cube_fell, cube_too_fast, nan kept; analysis only).
For every environment where the rule WOULD have fired, it classifies the event
(settle-flat / tip-away / other, by the palm-frame face tilt) and measures what happens
afterwards: is the cube still in the hand at 60 s, and does it keep rotating?

    python experiments/grasp_rotation_v2/scripts/analyze_settle.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_grasps import cube_tilt_in_palm  # noqa: E402

E = Path("experiments/grasp_rotation_v2")
DT = 0.05


def analyse(key):
  tr = np.load(E / "data" / f"{key}.npz")
  e, code = tr["end_step"], tr["end_code"]
  T, n = tr["yaw_inc"].shape
  held = tr["alive"].astype(bool)
  drift, tilt = tr["drift"].astype(np.float32), tr["tilt"].astype(np.float32)
  would = held & ((drift > 0.08) | (tilt > 0.8))
  would[:2] = False
  fired = would.any(0)
  t_star = np.where(fired, would.argmax(0), -1)
  q = tr["quat_palm"].astype(np.float32)
  pt0 = cube_tilt_in_palm(q[1])
  idx = np.where(fired)[0]
  pt_star = cube_tilt_in_palm(q[t_star[idx], idx])
  flat = pt_star < pt0[idx] - 0.2
  tip = pt_star > pt0[idx] + 0.2
  dropped = (e >= 0) & (code >= 2)  # remaining terminations: fell / too fast / nan
  yaw = tr["yaw_inc"].astype(np.float64)
  after = np.array([yaw[t_star[i]:, i].sum() / (2 * math.pi) for i in idx])
  dur_after = np.array([((e[i] if e[i] >= 0 else T) - t_star[i]) * DT for i in idx])
  res = {"episodes": int(n), "pose_rule_would_fire": float(fired.mean()),
         "would_fire_settle_flat": float(flat.mean()) if len(idx) else None,
         "would_fire_tip_away": float(tip.mean()) if len(idx) else None,
         "dropped_or_too_fast_60s_all": float(dropped.mean()),
         "turns60_without_pose_rule": float(yaw.sum(0).mean() / (2 * math.pi)),
         "turns30_without_pose_rule": float(yaw[:600].sum(0).mean() / (2 * math.pi))}
  for nm, m in (("settle_flat", flat), ("tip_away", tip), ("other", ~flat & ~tip)):
    if m.sum():
      ii = idx[m]
      res[nm] = {"n": int(m.sum()), "still_in_hand_at_60s": float(1 - dropped[ii].mean()),
                 "turns_after_event_mean": float(after[m].mean()),
                 "yaw_rate_after_event": float(after[m].sum() * 2 * math.pi / max(1e-9, dur_after[m].sum())),
                 "event_time_median_s": float(np.median(t_star[ii]) * DT),
                 "palm_tilt_start": float(pt0[ii].mean()), "palm_tilt_at_event": float(pt_star[m].mean()),
                 "drift_end_mm": float(1000 * drift[np.where(e[ii] >= 0, np.maximum(e[ii] - 1, 0), T - 1), ii].mean()),
                 "palm_tilt_end": float(cube_tilt_in_palm(q[np.where(e[ii] >= 0, np.maximum(e[ii] - 1, 0), T - 1), ii]).mean())}
  return res


def main():
  out = {}
  for p in ("A1", "R1"):
    for tag in ("bad", "all"):
      k = f"diag_noterm_{p}_{tag}_s6104"
      if (E / "data" / f"{k}.npz").exists():
        out[k] = analyse(k)
    k = f"diag_term_{p}_bad_s6104"
    if (E / "data" / f"{k}.npz").exists():
      tr = np.load(E / "data" / f"{k}.npz")
      e, code = tr["end_step"], tr["end_code"]
      out[k] = {"turns60_with_pose_rule": float(tr["yaw_inc"].astype(np.float64).sum(0).mean() / (2 * math.pi)),
                "turns30_with_pose_rule": float(tr["yaw_inc"][:600].astype(np.float64).sum(0).mean() / (2 * math.pi)),
                "failure60": float(((e >= 0) & (code >= 2)).mean())}
  (E / "results" / "termination_diagnostic.json").write_text(json.dumps(out, indent=1))
  print(json.dumps(out, indent=1))


if __name__ == "__main__":
  main()
