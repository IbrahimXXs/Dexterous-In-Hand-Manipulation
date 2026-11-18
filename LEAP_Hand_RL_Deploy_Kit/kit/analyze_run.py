#!/usr/bin/env python3
"""Compare estimated fingertip forces with the motor currents of a run.

For every step of a run log (monitor fields logged, or re-computed for older
logs) and every finger:
  F_est    learned force estimator (from proprioception only) [N]
  p        its contact probability
  I        sum of |current| of the finger's 4 motors [mA]
  F(I)     fingertip force implied by the currents: tau = Kt * I, least-squares
           J^T F = tau at the fingertip, component toward the estimated cube
           centre [N] (quasi-static; ignores gravity, friction, dynamics)

Reports, per finger: correlations r(F_est, I) and r(F_est, F(I)) (scale-free,
so Kt does not matter), the best lag, and the mean current when the estimator
says "contact" (p > 0.5) vs "no contact" (p < 0.2).  Plus estimated turns and
cube position.  Writes <log>_analysis.json and, if matplotlib is available,
<log>_analysis.png.

    uv run python LEAP_Hand_RL_Deploy_Kit/kit/analyze_run.py LEAP_Hand_RL_Deploy_Kit/logs/physical_X.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hand_scene import HandScene  # noqa: E402
from live_viewer import FINGER_MOTORS, KIT, _HW_TO_SERVER, load_log_packets  # noqa: E402
from monitor import FINGERS  # noqa: E402


def _r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 5 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _best_lag(x: np.ndarray, y: np.ndarray, max_lag: int = 5) -> tuple[int, float]:
    """Lag L (steps) maximising r(x[t], y[t+L]); positive = y follows x."""
    best = (0, _r(x, y))
    for L in range(-max_lag, max_lag + 1):
        if L == 0:
            continue
        a, b = (x[:-L], y[L:]) if L > 0 else (x[-L:], y[:L])
        r = _r(a, b)
        if not np.isnan(r) and (np.isnan(best[1]) or r > best[1]):
            best = (L, r)
    return best


def analyze(log: Path, monitor_dir: Path, kt: float) -> dict:
    pk, how = load_log_packets(log, monitor_dir)
    sc = HandScene()
    n = len(pk)
    F = np.array([p["monitor"]["force_n"] for p in pk])
    P = np.array([p["monitor"]["force_contact_prob"] for p in pk])
    cur = np.abs(np.array([p["current_ma_hw"] for p in pk]))
    I = np.stack([cur[:, FINGER_MOTORS[f]].sum(1) for f in FINGERS], 1)
    FI = np.zeros((n, 4))
    for i, p in enumerate(pk):
        sc.set_hand(np.asarray(p["q_server"]))
        sc.set_cube_palm(p["monitor"]["cube_pos_palm"], p["monitor"]["cube_quat_palm"])
        tau = kt * 1e-3 * np.asarray(p["current_ma_hw"], np.float64)[_HW_TO_SERVER]
        f = sc.tip_force_from_torque(tau, sc.cube_world()[0])
        FI[i] = [f[k] for k in FINGERS]
    t = np.array([p["t"] for p in pk])
    fingers = {}
    for k, f in enumerate(FINGERS):
        on, off = P[:, k] > 0.5, P[:, k] < 0.2
        lag, rlag = _best_lag(F[:, k], FI[:, k])
        fingers[f] = {
            "F_est_mean_N": float(F[:, k].mean()), "F_est_max_N": float(F[:, k].max()),
            "contact_fraction_p_gt_0.5": float(on.mean()),
            "r_Fest_vs_current": _r(F[:, k], I[:, k]),
            "r_Fest_vs_F_from_current": _r(F[:, k], FI[:, k]),
            "best_lag_steps_F_from_current": lag, "r_at_best_lag": rlag,
            "current_mA_when_contact": float(I[on, k].mean()) if on.any() else None,
            "current_mA_when_no_contact": float(I[off, k].mean()) if off.any() else None,
            "steps_contact": int(on.sum()), "steps_no_contact": int(off.sum()),
            "F_from_current_mean_N": float(FI[:, k].mean()),
        }
    pos = np.array([p["monitor"]["cube_pos_palm"] for p in pk])
    out = {
        "log": str(log), "steps": n, "duration_s": float(t[-1] - t[0]) if n else 0.0,
        "policy": pk[0]["policy"] if pk else None, "monitor_source": how, "kt_N_m_per_A": kt,
        "est_turns": float(pk[-1]["monitor"]["est_yaw_rad"]) / (2 * np.pi) if n else None,
        "est_cube_pos_palm_mean_mm": (1000 * pos.mean(0)).round(1).tolist(),
        "est_cube_pos_palm_std_mm": (1000 * pos.std(0)).round(1).tolist(),
        "fingers": fingers,
        "notes": [
            "F_est and contact probabilities are learned estimates from joint history (no force sensor).",
            "F(I) assumes quasi-static contact and Kt; it also contains gravity, friction and tracking effort, "
            "and the motors are current-limited (350 mA), so it is a cross-check, not ground truth.",
            "Positive est_turns = the direction the policy was trained to rotate (about the palm normal).",
        ],
    }
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
        for k, f in enumerate(FINGERS):
            a = ax[k]
            a.plot(t - t[0], F[:, k], label="F_est [N]", color="C0")
            a.plot(t - t[0], FI[:, k], label=f"F(I) [N], Kt={kt}", color="C3", alpha=0.7)
            a.fill_between(t - t[0], 0, P[:, k] * max(1.0, F[:, k].max()), color="C0", alpha=0.1, label="p_contact (scaled)")
            a2 = a.twinx()
            a2.plot(t - t[0], I[:, k], color="0.4", lw=0.8, label="|I| finger [mA]")
            a2.set_ylabel("mA")
            a.set_ylabel(f"{f}\nN")
            a.set_ylim(-1.0, max(3.0, 1.2 * float(F[:, k].max())))  # F(I) can be large near singular poses
            fd = fingers[f]
            a.set_title(f"{f}: r(F_est, I) = {fd['r_Fest_vs_current']:.2f}, r(F_est, F(I)) = "
                        f"{fd['r_Fest_vs_F_from_current']:.2f}", fontsize=9)
            if k == 0:
                a.legend(loc="upper left", fontsize=8)
                a2.legend(loc="upper right", fontsize=8)
        ax[-1].set_xlabel("time [s]")
        fig.suptitle(f"{log.name}  (est. turns {out['est_turns']:+.2f})")
        fig.tight_layout()
        png = log.with_name(log.stem + "_analysis.png")
        fig.savefig(png, dpi=110)
        out["figure"] = str(png)
    except ImportError:
        out["figure"] = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", type=Path, nargs="+")
    ap.add_argument("--monitor-dir", type=Path, default=KIT / "monitors")
    ap.add_argument("--kt", type=float, default=0.5, help="motor torque constant [N m / A] (approximate)")
    a = ap.parse_args()
    for log in a.logs:
        res = analyze(log, a.monitor_dir, a.kt)
        dest = log.with_name(log.stem + "_analysis.json")
        dest.write_text(json.dumps(res, indent=2) + "\n")
        print(f"\n{log.name}: {res['steps']} steps, policy {res['policy']}, est. turns {res['est_turns']:+.2f} "
              f"({res['monitor_source']})")
        print("  finger   F_est mean/max  contact%  r(F,I)  r(F,F(I))  lag  I contact/no-contact mA")
        for f, d in res["fingers"].items():
            ic, inc = d["current_mA_when_contact"], d["current_mA_when_no_contact"]
            print(f"  {f:7s} {d['F_est_mean_N']:5.2f}/{d['F_est_max_N']:4.2f}   {100 * d['contact_fraction_p_gt_0.5']:5.1f}  "
                  f"{d['r_Fest_vs_current']:+.2f}   {d['r_Fest_vs_F_from_current']:+.2f}    {d['best_lag_steps_F_from_current']:+d}  "
                  f"{'-' if ic is None else f'{ic:.0f}'} / {'-' if inc is None else f'{inc:.0f}'}")
        print(f"  -> {dest}" + (f", {res['figure']}" if res.get("figure") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
