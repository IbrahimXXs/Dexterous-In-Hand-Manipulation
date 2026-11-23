#!/usr/bin/env python3
"""C5-HW-V1 design support: does capping the simulated actuator at the deployed 350 mA reproduce
the hardware failure signature, and what does it cost in rotation?  Evaluation only (frozen
checkpoints, no training); reads sim_reference/sim_*.npz from sim_actuator_reference.py.

Signature metrics (hardware analogue in parentheses; LEAP_Hand_RL_Deploy_Kit/audit/20261120):
  sat_frac         share of joint-samples with |actuator force| >= 98% of its effort limit
                   (hardware: |current| >= 340 mA of the 350 mA limit)
  sustained_ge_Xs  share of environments with one joint saturated for >= X s without a break
                   (hardware: every C5 guard stop was preceded by >= 1 s of index saturation)
  blocked_frac     saturated AND |command - measured| >= 0.2 rad (the jam signature)
  turns30          cube yaw in 30 s / 2 pi (from the first step, environments that never terminated)

    python experiments/c5_hw_v1/scripts/sim_effort_diagnostic.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

E = Path(__file__).resolve().parents[1]
JOINTS_SERVER = ("if_mcp", "if_rot", "if_pip", "if_dip", "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
                 "rf_mcp", "rf_rot", "rf_pip", "rf_dip", "th_cmc", "th_axl", "th_mcp", "th_ipl")


def longest_run(mask: np.ndarray) -> np.ndarray:
  """Longest run of True along axis 0 for every column of a (T, ...) array."""
  best = np.zeros(mask.shape[1:], int)
  cur = np.zeros(mask.shape[1:], int)
  for m in mask:
    cur = np.where(m, cur + 1, 0)
    best = np.maximum(best, cur)
  return best


def analyse(path: Path) -> dict:
  d = np.load(path, allow_pickle=False)
  assert tuple(str(x) for x in d["joint_names"]) == JOINTS_SERVER
  q, c, F, L, al = d["q"], d["cmd"], d["force"], d["effort_limit"], d["alive"]
  dt = float(d["dt"])
  sat = np.abs(F) >= 0.98 * L[None]
  err = np.zeros_like(q)
  err[1:] = np.abs(c[:-1] - q[1:])
  blocked = sat & (err >= 0.2)
  lr = longest_run(sat) * dt  # (n_env, 16) seconds
  lr_any = lr.max(1)
  idx = [JOINTS_SERVER.index(j) for j in ("if_mcp", "if_pip")]
  survived = al[-1].astype(bool)
  turns = d["yaw_inc"].astype(np.float64).sum(0) / (2 * math.pi) if "yaw_inc" in d.files else np.full(q.shape[1], np.nan)
  alive_mask = al.astype(bool)
  return {"file": path.name, "policy": str(d["policy"]), "effort_limit_nm": [float(L.min()), float(L.max())],
          "override": [float(x) for x in d["effort_limit_override"]] if "effort_limit_override" in d.files else None,
          "n_envs": int(q.shape[1]), "steps": int(q.shape[0]), "survived_frac": float(survived.mean()),
          "turns30_mean": float(np.nanmean(turns)), "turns30_mean_survivors": float(np.nanmean(turns[survived])),
          "sat_frac_all": float(sat[alive_mask].mean()),
          "sat_frac_per_joint": {j: float(sat[..., k][alive_mask].mean()) for k, j in enumerate(JOINTS_SERVER)},
          "sat_frac_index_mcp_pip": float(sat[..., idx].any(-1)[alive_mask].mean()),
          "blocked_frac_all": float(blocked[alive_mask].mean()),
          "blocked_frac_index_mcp_pip": float(blocked[..., idx].any(-1)[alive_mask].mean()),
          "envs_sustained_ge_0.5s": float(np.mean(lr_any >= 0.5)), "envs_sustained_ge_1s": float(np.mean(lr_any >= 1.0)),
          "envs_sustained_ge_2s": float(np.mean(lr_any >= 2.0)),
          "envs_index_sustained_ge_1s": float(np.mean(lr[:, idx].max(1) >= 1.0)),
          "longest_sat_s_p90": float(np.percentile(lr_any, 90)),
          "joint_most_often_sustained": JOINTS_SERVER[int(np.argmax((lr >= 1.0).sum(0)))] if (lr >= 1.0).any() else None,
          "tracking_median_rad": float(np.median(err[1:][alive_mask[1:]]))}


def main() -> None:
  out = {}
  for p in sorted((E / "sim_reference").glob("sim_*.npz")):
    r = analyse(p)
    out[p.stem] = r
    print(f"{p.stem:40s} limit {r['effort_limit_nm'][0]:.3f}-{r['effort_limit_nm'][1]:.3f} Nm  turns30 {r['turns30_mean']:.3f}  "
          f"survived {r['survived_frac']:.3f}  sat {r['sat_frac_all']:.4f}  idx-sat {r['sat_frac_index_mcp_pip']:.4f}  "
          f"blocked {r['blocked_frac_all']:.4f}  envs>=1s {r['envs_sustained_ge_1s']:.3f}  idx>=1s {r['envs_index_sustained_ge_1s']:.3f}  "
          f"most {r['joint_most_often_sustained']}")
  (E / "results").mkdir(exist_ok=True)
  (E / "results" / "sim_effort_diagnostic.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
  main()
