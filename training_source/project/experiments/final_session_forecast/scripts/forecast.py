#!/usr/bin/env python3
"""Forecast for the final real-hand session (docs/hardware_protocol_final_DRAFT.md), from existing data only.

Written and hashed BEFORE the session; scored afterwards against the session's results without editing.
Arms: M400, A3, C5s3 (12 attempts each).  Quantities per arm, as the protocol defines them:
  rate   camera revolutions per minute while held; per-arm MEAN over attempts held >= 10 s (primary metric)
  tpc    quarter turns per finger cycle = camera quarter turns / encoder cycles, window 3 s -> drop;
         per-arm ratio of sums over attempts held >= 20 s (gait.py counter, unchanged)
  rhythm finger rhythm (FFT peak of PC1, Hz), window 3 s -> drop; per-arm MEDIAN over attempts held >= 20 s
  held60 share of attempts held 60 s (no drop, no stop)

Inputs (all in git, read only):
  LEAP_Hand_RL_Deploy_Kit/logs/trials.csv (20 Nov), trials_20261124.csv, trials_20261125.csv: operator quarter
    turns, drop time, stop reason, intervened; the run logs physical_*.npz (q_hw, elapsed_s) for the counter;
  LEAP_Hand_RL_Deploy_Kit/logs/trials_20261125_vision.csv: camera vs operator quarter turns (25 Nov);
  training_source/project/experiments/gait_efficiency/results/real_hand.json: camera_check_25nov (camera quarter
    turns inside the counting window) and the held-variant per-run rows (cross-check of the recomputed counter).

Method (B = 20,000 simulated sessions per arm and quantity):
  1. prior attempts of the arm = its rows in the three trial CSVs (counted quantities: count present, not intervened);
  2. Bayesian bootstrap: Dirichlet(1,...,1) weights over the prior attempts, then 12 new attempts drawn with these
     weights (each keeps its held time, so the share passing an inclusion rule is drawn too);
  3. small-n smoothing: each drawn attempt's value x exp(N(0, h^2)), h = s_run * n^(-1/5) (Silverman's n-rate),
     s_run = pooled within-session per-attempt SD of log(value) over every policy-session group of the three trial
     CSVs with >= 2 positive eligible values (all six policies; zeros cannot enter a log SD), n = prior attempts
     eligible for that quantity;
  4. camera conversion (rate, tpc): each attempt x a log-normal factor fitted to the 25 Nov attempts with both
     counts held >= 10 s (trials 1 and 5): rate: camera quarter turns to the drop / operator count;
     tpc: camera quarter turns inside the window 3 s -> drop / operator count over the whole hold;
  5. session shift: the arm statistic x exp(N(0, s_sess^2)), s_sess = RMS of the log ratio between two sessions of
     the same policy (C5s1 20 vs 24 Nov, C5s3 24 vs 25 Nov) / sqrt(2) (includes sampling noise: conservative);
  6. point forecast = median of the simulated statistic; interval = 5th-95th percentile (90%).
  held60: Jeffreys beta-binomial, Beta(k + 1/2, n - k + 1/2) for the share, then 12 attempts; median and 90% interval.
Assumptions stated in docs/forecast_final_session.md.

    uv run --no-project --with numpy --with scipy python \
        training_source/project/experiments/final_session_forecast/scripts/forecast.py
Writes results/forecast.json and results/forecast_tables.md.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

F = Path(__file__).resolve().parents[1]
PROJECT = F.parents[1]
ROOT = PROJECT.parents[1]
GE = PROJECT / "experiments" / "gait_efficiency"
LOGS = ROOT / "LEAP_Hand_RL_Deploy_Kit" / "logs"

_spec = importlib.util.spec_from_file_location("gait_real_hand", GE / "scripts" / "real_hand.py")
RH = importlib.util.module_from_spec(_spec)
sys.modules["gait_real_hand"] = RH
_spec.loader.exec_module(RH)
G = RH.G

ARMS = ("M400", "A3", "C5s3")
C5 = ("C5s1", "C5s3")
M_NEW = 12
B = 20000
MIN_RATE_S, MIN_GAIT_S = 10.0, 20.0
Z = (5, 95)


def load() -> list[dict]:
  runs = RH.load_runs()
  stop = {}
  for session, fname in RH.SESSIONS:
    for r in csv.DictReader(open(LOGS / fname, newline="")):
      stop[(session, int(r["trial"]))] = r["stop_reason"].strip()
  out = []
  for r in runs:
    r = dict(r)
    r["stop_reason"] = stop[(r["session"], r["trial"])]
    r["held60"] = (not r["dropped"]) and r["stop_reason"] == "max_steps" and not r["intervened"]
    r["counted"] = (r["qturns"] is not None) and not r["intervened"]
    r["rate"] = r["qturns"] / 4.0 / (r["held_s"] / 60.0) if r["counted"] and r["held_s"] > 0 else None
    if (not r["intervened"]) and r["held_s"] >= MIN_GAIT_S:
      c = G.count_cycles(r["q"], r["t"], t_end_s=r["held_s"])
      r["cycles"], r["rhythm"] = (c["cycles"], c["rhythm_hz"]) if c["status"] == "ok" else (None, None)
    else:
      r["cycles"] = r["rhythm"] = None
    del r["q"], r["t"]
    out.append(r)
  return out


def crosscheck(runs: list[dict]) -> int:
  ref = {(x["session"], x["trial"]): x for x in json.loads((GE / "results" / "real_hand.json").read_text())["variants"]["held"]["per_run"]}
  n = 0
  for r in runs:
    k = (r["session"], r["trial"])
    if k in ref and r["cycles"] is not None:
      assert abs(ref[k]["cycles"] - r["cycles"]) < 1e-9 and abs(ref[k]["rhythm_hz"] - r["rhythm"]) < 1e-12, k
      n += 1
  return n


def pooled_log_sd(groups: list[list[float]]) -> float:
  num = den = 0.0
  for g in groups:
    v = np.log(np.asarray([x for x in g if x and x > 0]))
    if len(v) >= 2:
      num += (len(v) - 1) * v.var(ddof=1)
      den += len(v) - 1
  return float(np.sqrt(num / den))


def stat_rate(rs):
  v = [r["rate"] for r in rs if r["counted"] and r["held_s"] >= MIN_RATE_S]
  return float(np.mean(v)) if v else None


def stat_tpc(rs):
  e = [r for r in rs if r["counted"] and r["cycles"]]
  return float(sum(r["qturns"] for r in e) / sum(r["cycles"] for r in e)) if e else None


def stat_rhythm(rs):
  v = [r["rhythm"] for r in rs if r["rhythm"] is not None]
  return float(np.median(v)) if v else None


def dispersion_inputs(runs: list[dict]) -> dict:
  by = {}
  for r in runs:
    by.setdefault((r["policy"], r["session"]), []).append(r)
  rate_g = [[r["rate"] for r in g if r["counted"] and r["held_s"] >= MIN_RATE_S] for g in by.values()]
  tpc_g = [[r["qturns"] / r["cycles"] for r in g if r["counted"] and r["cycles"]] for g in by.values()]
  rhy_g = [[r["rhythm"] for r in g if r["rhythm"] is not None] for g in by.values()]
  pairs = ((("C5s1", "20 Nov"), ("C5s1", "24 Nov")), (("C5s3", "24 Nov"), ("C5s3", "25 Nov")))
  sess = {}
  for name, fn in (("rate", stat_rate), ("tpc", stat_tpc), ("rhythm", stat_rhythm)):
    d = [float(np.log(fn(by[b]) / fn(by[a]))) for a, b in pairs]
    sess[name] = {"log_ratios": d, "s_sess": float(np.sqrt(np.mean(np.square(d)) / 2))}
  return {"s_run": {"rate": pooled_log_sd(rate_g), "tpc": pooled_log_sd(tpc_g), "rhythm": pooled_log_sd(rhy_g)},
          "session": sess,
          "groups": {f"{p} {s}": {"rate": stat_rate(g), "tpc": stat_tpc(g), "rhythm": stat_rhythm(g),
                                  "n_rate": len([r for r in g if r["counted"] and r["held_s"] >= MIN_RATE_S]),
                                  "n_gait": len([r for r in g if r["counted"] and r["cycles"]])}
                     for (p, s), g in sorted(by.items())}}


def camera_inputs() -> dict:
  vis = {int(r["trial"]): r for r in csv.DictReader(open(LOGS / "trials_20261125_vision.csv", newline=""))}
  cam = json.loads((GE / "results" / "real_hand.json").read_text())["camera_check_25nov"]
  rate, tpc = {}, {}
  for t in (1, 5):  # the 25 Nov attempts with both counts and held >= 10 s (trial 4 held 5 s; 2 and 3 had no camera)
    v = vis[t]
    op = float(v["observed_quarter_turns"])
    camq = float(v["vision_qt_until_drop"] or v["vision_quarter_turns"])
    rate[t] = camq / op
    tpc[t] = cam[f"trial{t}"]["camera_qturns_in_window"] / op
  out = {}
  for k, d in (("rate", rate), ("tpc", tpc)):
    lv = np.log(np.array(list(d.values())))
    out[k] = {"ratios": {f"trial{t}": float(x) for t, x in d.items()}, "mu_log": float(lv.mean()), "sd_log": float(lv.std(ddof=1))}
  return out


def simulate(arm_runs: list[dict], kind: str, h: float, s_sess: float, camf: dict | None, rng) -> dict:
  pop = [r for r in arm_runs if r["counted"]] if kind != "rhythm" else [r for r in arm_runs if not r["intervened"]]
  n = len(pop)
  res = np.full(B, np.nan)
  for b in range(B):
    w = rng.dirichlet(np.ones(n))
    draw = [pop[i] for i in rng.choice(n, M_NEW, p=w)]
    if kind == "rate":
      e = [r for r in draw if r["held_s"] >= MIN_RATE_S]
      if not e:
        continue
      x = np.array([r["rate"] for r in e]) * np.exp(rng.normal(0, h, len(e))) * np.exp(rng.normal(camf["mu_log"], camf["sd_log"], len(e)))
      s = x.mean()
    elif kind == "tpc":
      e = [r for r in draw if r["cycles"]]
      if not e:
        continue
      q = np.array([r["qturns"] for r in e]) * np.exp(rng.normal(0, h, len(e))) * np.exp(rng.normal(camf["mu_log"], camf["sd_log"], len(e)))
      s = q.sum() / sum(r["cycles"] for r in e)
    else:
      e = [r for r in draw if r["rhythm"] is not None]
      if not e:
        continue
      s = float(np.median(np.array([r["rhythm"] for r in e]) * np.exp(rng.normal(0, h, len(e)))))
    res[b] = s * np.exp(rng.normal(0, s_sess))
  ok = res[np.isfinite(res)]
  return {"point_median": float(np.median(ok)), "interval90": [float(np.percentile(ok, Z[0])), float(np.percentile(ok, Z[1]))],
          "p_undefined": float(1 - ok.size / B)}


def held60(arm_runs: list[dict], rng) -> dict:
  n, k = len(arm_runs), sum(r["held60"] for r in arm_runs)
  p = rng.beta(k + 0.5, n - k + 0.5, B)
  kn = rng.binomial(M_NEW, p)
  lo, md, hi = np.percentile(kn, [Z[0], 50, Z[1]])
  return {"prior_k": int(k), "prior_n": n, "point_of_12": int(md), "interval90_of_12": [int(lo), int(hi)],
          "point_share": float(md / M_NEW), "interval90_share": [float(lo / M_NEW), float(hi / M_NEW)]}


def main() -> None:
  rng = np.random.default_rng(20261127)
  runs = load()
  ncheck = crosscheck(runs)
  c5 = dispersion_inputs(runs)
  cam = camera_inputs()
  out = {"label": "FORECAST, written before the final session from existing data only; do not edit after hashing",
         "m_new_attempts_per_arm": M_NEW, "B": B, "interval": "90% (5th-95th percentile)", "rng_seed": 20261127,
         "crosscheck_rows_equal_real_hand_json": ncheck, "dispersion_inputs": c5, "camera_inputs": cam, "arms": {}}
  for arm in ARMS:
    ar = [r for r in runs if r["policy"] == arm]
    prior = {"attempts": [{k: r[k] for k in ("session", "trial", "held_s", "qturns", "dropped", "stop_reason", "intervened",
                                             "held60", "rate", "cycles", "rhythm")} for r in ar],
             "n_attempts": len(ar), "n_rate_eligible": sum(1 for r in ar if r["counted"] and r["held_s"] >= MIN_RATE_S),
             "n_gait_eligible_counted": sum(1 for r in ar if r["counted"] and r["cycles"]),
             "n_rhythm_eligible": sum(1 for r in ar if r["rhythm"] is not None),
             "rate_operator_mean": stat_rate(ar), "tpc_operator": stat_tpc(ar), "rhythm_median": stat_rhythm(ar)}
    fc = {}
    for kind, nkey in (("rate", "n_rate_eligible"), ("tpc", "n_gait_eligible_counted"), ("rhythm", "n_rhythm_eligible")):
      h = c5["s_run"][kind] * prior[nkey] ** (-0.2)
      fc[kind] = {"h": h, **simulate(ar, kind, h, c5["session"][kind]["s_sess"], cam.get(kind), rng)}
    fc["held60"] = held60(ar, rng)
    out["arms"][arm] = {"prior": prior, "forecast": fc}
  (F / "results" / "forecast.json").write_text(json.dumps(out, indent=1) + "\n")

  L = ["| Arm | Camera rev/min while held (mean, held ≥ 10 s) | Quarter turns per finger cycle (camera, held ≥ 20 s) | "
       "Finger rhythm Hz (median, held ≥ 20 s) | Attempts held 60 s (of 12) |", "|---|---|---|---|---|"]
  for arm in ARMS:
    f = out["arms"][arm]["forecast"]
    cell = lambda k, d: f"{f[k]['point_median']:.{d}f} [{f[k]['interval90'][0]:.{d}f}, {f[k]['interval90'][1]:.{d}f}]"
    L.append(f"| {arm} | {cell('rate', 2)} | {cell('tpc', 2)} | {cell('rhythm', 2)} | "
             f"{f['held60']['point_of_12']} [{f['held60']['interval90_of_12'][0]}, {f['held60']['interval90_of_12'][1]}] |")
  (F / "results" / "forecast_tables.md").write_text("\n".join(L) + "\n")
  print("\n".join(L))
  print(json.dumps({"c5": {k: c5[k] for k in ("s_run", "session")}, "camera": cam}, indent=1))
  for arm in ARMS:
    p = out["arms"][arm]["prior"]
    print(arm, {k: p[k] for k in p if k != "attempts"}, {k: (v.get("h"), v.get("p_undefined")) for k, v in out["arms"][arm]["forecast"].items()})


if __name__ == "__main__":
  main()
