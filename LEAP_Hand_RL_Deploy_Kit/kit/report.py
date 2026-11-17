#!/usr/bin/env python3
"""Summarise a shadow / physical run log (.npz) written by deploy_direct.py."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np

HW = ("if_rot", "if_mcp", "if_pip", "if_dip", "mf_rot", "mf_mcp", "mf_pip", "mf_dip",
      "rf_rot", "rf_mcp", "rf_pip", "rf_dip", "th_cmc", "th_axl", "th_mcp", "th_ipl")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", nargs="?", type=Path, default=None, help="default: newest file in logs/")
    args = ap.parse_args()
    path = args.log
    if path is None:
        files = sorted(glob.glob(str(Path(__file__).resolve().parent.parent / "logs" / "*.npz")), key=os.path.getmtime)
        if not files:
            print("no logs yet")
            return 1
        path = Path(files[-1])
    d = np.load(path, allow_pickle=False)
    n = len(d["elapsed_s"])
    print(f"log: {path}")
    print(f"started {d['started_at']}, completed={bool(d['completed'])}, stop_reason={d['stop_reason']}")
    print(f"steps={n}, duration={d['elapsed_s'][-1] - d['elapsed_s'][0]:.1f} s, "
          f"loop compute mean/max = {1e3 * d['loop_compute_s'].mean():.1f}/{1e3 * d['loop_compute_s'].max():.1f} ms")
    cmd, q, cur = d["command_hw"], d["q_hw"], d["current_ma"]
    temp = d["temperature_c"]
    step = np.abs(np.diff(cmd, axis=0)) if n > 1 else np.zeros((1, 16))
    print(f"max |current| {np.abs(cur).max():.0f} mA, max temp {np.nanmax(temp):.0f} C, "
          f"max tracking error {d['tracking_error_rad'].max():.3f} rad")
    print("\njoint     cmd_min  cmd_max  q_min    q_max   max_step  max_mA")
    for j, name in enumerate(HW):
        print(f"{name:8s} {cmd[:, j].min():7.3f} {cmd[:, j].max():7.3f} {q[:, j].min():7.3f} "
              f"{q[:, j].max():7.3f} {step[:, j].max():8.4f} {np.abs(cur[:, j]).max():7.0f}")
    print(f"\nstall recoveries: {int(d['stall_recovery_count'].max())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
