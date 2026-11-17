#!/usr/bin/env python3
"""Inspect a policy folder and run it open-loop without hardware.

Checks that the checkpoint loads, that its input/output sizes match the task
spec, and runs N steps against an ideal "perfect tracking" hand to make sure
the commands are finite, inside limits, and not frozen.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_runtime import LeapPolicy, SERVER_JOINT_NAMES, export_npz  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("policy", type=Path, help="folder with model_*.pt (+ env.yaml, agent.yaml)")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--export-npz", type=Path, default=None)
    args = ap.parse_args()

    try:
        policy = LeapPolicy(args.policy)
    except Exception as exc:  # show a clean message
        print(f"\n[inspect] POLICY NOT USABLE: {exc}\n")
        return 1
    info = policy.describe()
    for key, value in info.items():
        if key == "notes":
            for note in value:
                print(f"NOTE: {note}")
        else:
            print(f"{key:18s}: {value}")

    q = policy.default_joint_pos.copy()
    policy.reset(q)
    targets, raws = [], []
    for _ in range(args.steps):
        t, r = policy.step(q)
        targets.append(t)
        raws.append(r)
        q = 0.7 * q + 0.3 * t  # crude first-order tracking hand
    targets = np.array(targets)
    raws = np.array(raws)
    ok = bool(np.all(np.isfinite(targets)) and np.all(np.isfinite(raws)))
    sat = float(np.mean(np.abs(raws) >= 1.0))
    travel = np.abs(np.diff(targets, axis=0)).sum(axis=0)
    print(f"\n[inspect] {args.steps} open-loop steps ({args.steps / policy.spec.policy_hz:.1f} s at {policy.spec.policy_hz:g} Hz)")
    print(f"[inspect] finite outputs: {ok}")
    print(f"[inspect] action saturation (|a|>=1): {sat * 100:.0f}% of entries")
    print("[inspect] total joint travel [rad]: " + ", ".join(
        f"{n}={v:.2f}" for n, v in zip(SERVER_JOINT_NAMES, travel)))
    if not ok:
        print("[inspect] FAIL: non-finite outputs")
        return 1
    if float(travel.max()) < 1e-3:
        print("[inspect] WARNING: commands never move - check the checkpoint/config pair")
    if args.export_npz:
        export_npz(policy, args.export_npz)
        print(f"[inspect] exported NumPy weights to {args.export_npz}")
    print("[inspect] OK - policy is loadable and produces valid commands")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
