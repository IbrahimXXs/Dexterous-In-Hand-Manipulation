#!/usr/bin/env python3
"""PT-student Step 5 (PLAN.md 4.2; CPU; EXPLORATORY): does S-TF's cube-orientation head follow the real cube?

1. Rebuild the observation of every real C5 seed 1 / seed 3 run with the kit's own code (LeapPolicy of the seed that
   ran, as in seed_jury/scripts/real_replay.py) and check that it reproduces the logged raw action (<= 1e-4);
   otherwise stop.
2. Feed the observations to S-TF; its orientation head gives yaw modulo 90 deg (4 psi = atan2(sin, cos)); unwrapped
   / 4 = cumulative yaw.
3. 25 Nov trials 1, 4, 5: compare with the camera's cumulative quarter turns from the policy start to the operator's
   drop (or the end): Pearson r of the yaw rates (1-s centred slope, on the camera's sample times) and cumulative
   quarter turns; each series' sign set so that its net rotation is positive.
Reference (same trials, same measures): estimator B's on-hand yaw (``mon_est_yaw_rad`` in the run logs; REPORT 7.4).
Extra (not in the plan; exploratory): across the C5 runs held >= 10 s, Spearman of predicted vs operator quarter turns
per held minute, for S-TF and for estimator B (rates: total counts would mostly measure run length).
These runs were driven by C5, not by the student.

    python experiments/pt_student/scripts/real_log.py [--student experiments/pt_student/runs/S-TF/final.pt]
Writes results/real_log.json and results/real_log.md.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

E = Path(__file__).resolve().parents[1]
PROJECT = E.parents[1]
ROOT = PROJECT.parents[1]
LOGS = ROOT / "LEAP_Hand_RL_Deploy_Kit" / "logs"


def _load(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[name] = mod
  spec.loader.exec_module(mod)
  return mod


def _stub_getattr(name: str):
  if name.startswith("__"):
    raise AttributeError(name)
  return object


if "dynamixel_sdk" not in sys.modules:  # import-only stub (no bus code runs); dunder lookups must fail for torch/inspect
  import types

  _stub = types.ModuleType("dynamixel_sdk")
  _stub.__getattr__ = _stub_getattr  # type: ignore[attr-defined]
  sys.modules["dynamixel_sdk"] = _stub
RR = _load("seed_jury_real_replay", PROJECT / "experiments" / "seed_jury" / "scripts" / "real_replay.py")
M = _load("pt_models", E / "scripts" / "models.py")
GATE = 1e-4


def replay_obs(run: dict, pol) -> tuple[np.ndarray, np.ndarray, float, dict]:
  d = np.load(LOGS / run["log"], allow_pickle=False)
  q_hw, cmd_hw, raw_log = d["q_hw"], d["command_hw"], d["raw_action_server"]
  el = np.asarray(d["elapsed_s"], dtype=np.float64)
  cmd0 = RR.DD._hw_to_server(cmd_hw[0])
  q_reset = (cmd0 - RR.delta(raw_log[0], pol.spec)).astype(np.float32)
  pol.reset(q_reset)
  obs, err = [], 0.0
  for k in range(len(el)):
    applied = q_reset if k == 0 else RR.DD._hw_to_server(cmd_hw[k - 1])
    _, raw = pol.step(RR.DD._hw_to_server(q_hw[k]), None, applied)
    err = max(err, float(np.max(np.abs(raw - raw_log[k]))))
    obs.append(pol.observation().copy())
  return el - el[0], np.stack(obs), err, {"est_yaw": np.asarray(d["mon_est_yaw_rad"], dtype=np.float64) if "mon_est_yaw_rad" in d.files else None}


def slope(t: np.ndarray, y: np.ndarray, at: np.ndarray, half: float = 0.5) -> np.ndarray:
  out = np.full(len(at), np.nan)
  for i, x in enumerate(at):
    m = np.abs(t - x) <= half
    if m.sum() >= 3 and np.ptp(t[m]) >= half:
      out[i] = np.polyfit(t[m], y[m], 1)[0]
  return out


def compare(t_pred: np.ndarray, qt_pred: np.ndarray, tc: np.ndarray, qc: np.ndarray, t_end: float) -> dict:
  m = (tc >= 0) & (tc <= t_end)
  tc, qc = tc[m], qc[m] - qc[m][0]
  mp = t_pred <= t_end
  tp, yp = t_pred[mp], qt_pred[mp] - qt_pred[mp][0]
  sp, sc = (1.0 if yp[-1] >= 0 else -1.0), (1.0 if qc[-1] >= 0 else -1.0)
  yp, qc = yp * sp, qc * sc
  rp, rc = slope(tp, yp, tc), slope(tc, qc, tc)
  ok = np.isfinite(rp) & np.isfinite(rc)
  r = float(np.corrcoef(rp[ok], rc[ok])[0, 1]) if ok.sum() >= 5 and np.std(rp[ok]) > 0 and np.std(rc[ok]) > 0 else None
  return {"window_s": [0.0, float(t_end)], "n_rate_samples": int(ok.sum()), "rate_r": r,
          "pred_qturns": float(yp[-1]), "camera_qturns": float(qc[-1]), "pred_sign_flipped": sp < 0, "camera_sign_flipped": sc < 0}


def spearman(x, y) -> float:
  rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
  return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
  import torch

  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--student", default=str(E / "runs" / "S-TF" / "final.pt"))
  ap.add_argument("--out", default=str(E / "results"))
  a = ap.parse_args()
  seeds = RR.load_seeds()
  student = M.load(a.student, "cpu")
  runs = RR.load_runs()
  out = {"label": "EXPLORATORY (PLAN 4.2): real C5-driven runs fed to S-TF; single training seed", "student": a.student, "gate_tol": GATE, "runs": []}
  per = {}
  for r in runs:
    t, obs, err, extra = replay_obs(r, seeds[r["seed"]]["policy"])
    with torch.no_grad():
      o = student(torch.as_tensor(obs, dtype=torch.float32))
    four_psi = torch.atan2(o["ori"][:, 0], o["ori"][:, 1]).numpy().astype(np.float64)
    yaw = np.unwrap(four_psi) / 4.0
    t_end = r["t_drop"] if r["dropped"] else float(t[-1])
    m = t <= t_end
    rec = {"session": r["session"], "trial": r["trial"], "seed": r["seed"], "gate_raw_max_err": err, "t_end": t_end,
           "operator_qturns": None, "pred_qturns_abs": float(abs(yaw[m][-1] - yaw[m][0]) / (np.pi / 2)),
           "pred_tilt_mean": float(np.hypot(o["ori"][:, 2], o["ori"][:, 3]).numpy()[m].mean())}
    if extra["est_yaw"] is not None:
      ey = extra["est_yaw"]
      rec["estB_qturns_abs"] = float(abs(ey[m][-1] - ey[m][0]) / (np.pi / 2))
    per[(r["session"], r["trial"])] = (t, yaw, extra["est_yaw"])
    out["runs"].append(rec)
  out["gate_raw_max_err"] = max(x["gate_raw_max_err"] for x in out["runs"])
  out["gate_passed"] = bool(out["gate_raw_max_err"] <= GATE)
  if not out["gate_passed"]:
    Path(a.out, "real_log.json").write_text(json.dumps(out, indent=1) + "\n")
    print("GATE FAILED: step skipped", out["gate_raw_max_err"])
    sys.exit(2)
  # operator counts (trial CSVs)
  ops = {}
  for session, fname in RR.SESSIONS:
    for row in csv.DictReader(open(LOGS / fname, newline="")):
      if row["observed_turns"].strip():
        ops[(session, int(row["trial"]))] = float(row["observed_turns"])
  for x in out["runs"]:
    x["operator_qturns"] = ops.get((x["session"], x["trial"]))
  # camera comparison, 25 Nov trials 1, 4, 5
  cam = {}
  for tr in (1, 4, 5):
    rows = list(csv.DictReader(open(LOGS / f"trials_20261125_trial{tr}_vision.csv", newline="")))
    tc = np.array([float(x["t_from_policy_start_s"]) for x in rows])
    qc = np.array([float(x["cumulative_quarter_turns"]) for x in rows])
    t, yaw, ey = per[("25 Nov", tr)]
    rr = next(x for x in out["runs"] if x["session"] == "25 Nov" and x["trial"] == tr)
    cam[f"trial{tr}"] = {"S-TF": compare(t, yaw / (np.pi / 2), tc, qc, rr["t_end"]),
                         "estimator_B": compare(t, ey / (np.pi / 2), tc, qc, rr["t_end"]) if ey is not None else None,
                         "operator_qturns": rr["operator_qturns"]}
  out["camera_25nov"] = cam
  ok = [x for x in out["runs"] if x["operator_qturns"] is not None]
  ok = [x for x in ok if x["t_end"] >= 10.0]  # rates over holds >= 10 s (counts over shorter holds are mostly placement)
  rate = lambda x, k: x[k] / (x["t_end"] / 60.0)  # noqa: E731  quarter turns per held minute (removes the duration confound)
  out["operator_spearman"] = {"n_runs": len(ok), "measure": "quarter turns per held minute, runs held >= 10 s",
                              "S-TF": spearman([rate(x, "pred_qturns_abs") for x in ok], [rate(x, "operator_qturns") for x in ok]),
                              "estimator_B": spearman([rate(x, "estB_qturns_abs") for x in ok], [rate(x, "operator_qturns") for x in ok])
                              if all("estB_qturns_abs" in x for x in ok) else None}
  Path(a.out).mkdir(parents=True, exist_ok=True)
  Path(a.out, "real_log.json").write_text(json.dumps(out, indent=1) + "\n")
  f = lambda x, d=2: "–" if x is None else f"{x:.{d}f}"  # noqa: E731
  L = ["# PT-student Step 5: S-TF's cube-yaw head on the real C5 runs (EXPLORATORY)", "",
       f"Generated by `scripts/real_log.py`. **{out['label']}.** Observations rebuilt with the kit's code; the logged policy reproduces its "
       f"raw actions to {out['gate_raw_max_err']:.1e} (gate {GATE:.0e}: {'passed' if out['gate_passed'] else 'FAILED'}), {len(out['runs'])} runs.", "",
       "## 25 Nov camera trials (policy start → drop or end)", "",
       "| Trial | Window s | Camera quarter turns | Operator | S-TF quarter turns | S-TF rate r | Estimator B quarter turns | B rate r |", "|---|---|---|---|---|---|---|---|"]
  for k, v in cam.items():
    s, b = v["S-TF"], v["estimator_B"] or {}
    L.append(f"| {k} | 0–{s['window_s'][1]:g} | {f(s['camera_qturns'],1)} | {f(v['operator_qturns'],0)} | {f(s['pred_qturns'],1)} | {f(s['rate_r'])} | "
             f"{f(b.get('pred_qturns'),1)} | {f(b.get('rate_r'))} |")
  sp = out["operator_spearman"]
  L += ["", "Signs: each series is flipped so that its net rotation is positive (the camera axis sign is set by one trial; yaw from joints is modulo 90°).", "",
        f"## All C5 runs vs the operator's count (extra, not in the plan)", "",
        f"Spearman of |predicted| vs operator quarter turns per held minute (runs held ≥ 10 s; rates remove the run-length confound), {sp['n_runs']} runs: "
        f"S-TF {f(sp['S-TF'])}, estimator B {f(sp['estimator_B'])}."]
  Path(a.out, "real_log.md").write_text("\n".join(L) + "\n")
  print("\n".join(L))


if __name__ == "__main__":
  main()
