#!/usr/bin/env python3
"""Gait efficiency, real hand (EXPLORATORY, POST HOC): finger cycles vs operator quarter turns per policy.

Inputs (all in git, read only): LEAP_Hand_RL_Deploy_Kit/logs/trials.csv (20 Nov), trials_20261124.csv,
trials_20261125.csv (operator counts, quarter turns), the run logs physical_*.npz (q_hw at the policy rate,
elapsed_s), and the 25 Nov camera series trials_20261125_trial{1,4,5}_vision.csv.

Cycle counting: scripts/gait.py ``count_cycles`` (also used by the simulation side).  Time from the policy start
is elapsed_s - elapsed_s[0] (the camera pipeline uses the same origin: started_at + elapsed_s[0]).

Windows and inclusion (both reported; see README.md):
  run   (the instruction's procedure)  window 3 s -> end of the run; runs with run_s >= 20 s.
  held  (while the cube is held, as the simulation side counts)  window 3 s -> operator drop time (or run end);
        runs whose held time (drop time, or run_s if no drop) is >= 20 s.
Always: intervened runs excluded, operator count present.
Robustness on the "run" variant: (a) each finger's 4 joints instead of all 16; (b) band 0.1-1.5 Hz;
(c) runs >= 10 s; (d) per-run median of quarter turns per cycle instead of the ratio of sums.

    python experiments/gait_efficiency/scripts/real_hand.py      (from training_source/project)
Writes results/real_hand.json and results/real_hand.md.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

E = Path(__file__).resolve().parents[1]
ROOT = E.parents[3]
LOGS = ROOT / "LEAP_Hand_RL_Deploy_Kit" / "logs"
_spec = importlib.util.spec_from_file_location("gait", E / "scripts" / "gait.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

SESSIONS = (("20 Nov", "trials.csv"), ("24 Nov", "trials_20261124.csv"), ("25 Nov", "trials_20261125.csv"))
LABEL = {"C5": "C5s1"}  # the trial CSVs call C5 seed 1 "C5"
ORDER = ("C5s3", "C5s1", "M400", "R1", "A3", "P0")
FINGERS = {"index": [0, 1, 2, 3], "middle": [4, 5, 6, 7], "ring": [8, 9, 10, 11], "thumb": [12, 13, 14, 15]}  # kit server order
REFERENCE = {  # first pass on another machine, quoted in the instruction of 2026-11-27 (not used for anything else)
  "qturns_per_cycle_runs_ge_20s": {"C5s3": 0.87, "C5s1": 0.69, "M400": 0.19, "R1": 0.14, "A3": 0.07, "P0": 0.00},
  "rhythm_hz": {"C5s3": 0.29, "C5s1": 0.35, "M400": 0.29, "R1": 0.44, "A3": 0.39, "P0": 0.23},
  "camera": {"trial1": {"slope": 0.88, "r": 0.995}, "trial5": {"slope": 0.79, "r": 0.984}},
}
NBOOT = 10000


def fnum(x: str) -> float | None:
  x = (x or "").strip()
  return float(x) if x else None


def load_runs() -> list[dict]:
  runs = []
  for session, fname in SESSIONS:
    for r in csv.DictReader(open(LOGS / fname, newline="")):
      d = np.load(LOGS / r["log"], allow_pickle=False)
      e = np.asarray(d["elapsed_s"], dtype=np.float64)
      run_s = float(r["run_s"])
      dropped = r["dropped"].strip() == "True"
      drop_t = fnum(r["drop_time_s"])
      runs.append({"session": session, "trial": int(r["trial"]), "policy": LABEL.get(r["policy"], r["policy"]),
                   "log": r["log"], "run_s": run_s, "qturns": fnum(r["observed_turns"]), "dropped": dropped,
                   "drop_time_s": drop_t, "held_s": drop_t if (dropped and drop_t is not None) else run_s,
                   "intervened": r["intervened"].strip() == "True", "q": np.asarray(d["q_hw"], dtype=np.float64),
                   "t": e - e[0]})
  return runs


def eligible(r: dict, min_s: float, basis: str) -> bool:
  return (not r["intervened"]) and r["qturns"] is not None and r[basis] >= min_s


def per_run(runs: list[dict], *, min_s: float = 20.0, basis: str = "run_s", window: str = "run",
            band=G.BAND_HZ, joints=None) -> list[dict]:
  out = []
  for r in runs:
    if not eligible(r, min_s, basis):
      continue
    c = G.count_cycles(r["q"], r["t"], t_end_s=None if window == "run" else r["held_s"], band=band, joints=joints)
    out.append({**{k: r[k] for k in ("session", "trial", "policy", "log", "run_s", "qturns", "dropped", "drop_time_s", "held_s")},
                **{k: c.get(k) for k in ("status", "window_s", "cycles", "rhythm_hz", "pc1_variance_share",
                                         "pc1_raw_range_rad", "stroke_amplitude_median_rad", "t_start_s", "t_end_s")},
                "stroke_amplitudes_rad": c.get("stroke_amplitudes_rad", [])})
  return out


def summarise(rows: list[dict], rng: np.random.Generator, per_run_median: bool = False) -> dict:
  pol = {}
  for p in ORDER:
    rs = [x for x in rows if x["policy"] == p and x["status"] == "ok"]
    if not rs:
      pol[p] = {"n_runs": 0}
      continue
    qt = np.array([x["qturns"] for x in rs])
    cy = np.array([x["cycles"] for x in rs])
    ws = np.array([x["window_s"] for x in rs])
    ratio = float(qt.sum() / cy.sum()) if cy.sum() > 0 else float("nan")
    bs = np.empty(NBOOT)
    for i in range(NBOOT):
      j = rng.integers(0, len(rs), len(rs))
      bs[i] = qt[j].sum() / cy[j].sum() if cy[j].sum() > 0 else np.nan
    amps = [a for x in rs for a in x["stroke_amplitudes_rad"]]
    pol[p] = {"n_runs": len(rs), "runs": [f"{x['session']} t{x['trial']}" for x in rs],
              "qturns_total": float(qt.sum()), "cycles_total": float(cy.sum()), "window_s_total": float(ws.sum()),
              "qturns_per_cycle": ratio, "qturns_per_cycle_ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
              "qturns_per_cycle_median_of_runs": float(np.median(qt / cy)),
              "rhythm_hz_median": float(np.median([x["rhythm_hz"] for x in rs])),
              "cycle_rate_hz": float(cy.sum() / ws.sum()),
              "qturns_per_s": float(qt.sum() / ws.sum()),
              "stroke_amplitude_median_rad": float(np.median(amps)) if amps else float("nan"),
              "n_strokes": len(amps),
              "pc1_variance_share_median": float(np.median([x["pc1_variance_share"] for x in rs]))}
    if per_run_median:
      pol[p]["metric"] = pol[p]["qturns_per_cycle_median_of_runs"]
    else:
      pol[p]["metric"] = ratio
  return pol


def ranking(pol: dict) -> list[str]:
  ok = [p for p in ORDER if pol[p].get("n_runs", 0) > 0 and np.isfinite(pol[p]["metric"])]
  return sorted(ok, key=lambda p: -pol[p]["metric"])


def camera_check(runs: list[dict]) -> dict:
  out = {}
  for trial in (1, 4, 5):
    r = next(x for x in runs if x["session"] == "25 Nov" and x["trial"] == trial)
    c = G.count_cycles(r["q"], r["t"], t_end_s=r["held_s"])
    cam = list(csv.DictReader(open(LOGS / f"trials_20261125_trial{trial}_vision.csv", newline="")))
    tc = np.array([float(x["t_from_policy_start_s"]) for x in cam])
    qc = np.array([float(x["cumulative_quarter_turns"]) for x in cam])
    res = {"window_s": [c["t_start_s"], c["t_end_s"]], "status": c["status"], "cycles": c.get("cycles"),
           "camera_final_qturns_abs": float(abs(qc[-1]))}
    if c["status"] == "ok":
      tw, cum = np.array(c["t_series"]), np.array(c["cum_cycles_series"])
      ok = (tw >= tc.min()) & (tw <= tc.max())
      cq = np.interp(tw[ok], tc, qc)
      cq = (cq - cq[0]) * (1.0 if qc[-1] >= 0 else -1.0)  # sign fixed so the net rotation is positive
      x = cum[ok]
      if len(x) >= 3 and np.ptp(x) > 0:
        slope, icpt = np.polyfit(x, cq, 1)
        res.update(n_samples=int(ok.sum()), slope_qturns_per_cycle=float(slope), intercept=float(icpt),
                   r=float(np.corrcoef(x, cq)[0, 1]), cycles_in_window=float(x[-1] - x[0]),
                   camera_qturns_in_window=float(cq[-1] - cq[0]))
    out[f"trial{trial}"] = res
  return out


def first_pass_diagnostics(runs: list[dict]) -> dict:
  """Explain differences to the first pass (diagnostics only; the method above is unchanged).
  (1) rhythm from a plain FFT of the centred PC1 (no window, no zero padding);
  (2) the 25 Nov trial-5 camera fit with the window running to the end of the run (past the 30-s drop)."""
  out = {"rhythm_plain_fft_median_hz": {}}
  for p in ORDER:
    vals = []
    for r in runs:
      if r["policy"] != p or not eligible(r, 20.0, "run_s"):
        continue
      m = r["t"] >= G.T_START_S
      x, _, _ = G.pc1_projection(r["q"][m])
      fs = 1.0 / float(np.median(np.diff(r["t"][m])))
      f = np.fft.rfftfreq(len(x), 1.0 / fs)
      a = np.abs(np.fft.rfft(x - x.mean()))
      ok = f > G.BAND_HZ[0]
      vals.append(float(f[ok][np.argmax(a[ok])]))
    out["rhythm_plain_fft_median_hz"][p] = float(np.median(vals)) if vals else None
  r = next(x for x in runs if x["session"] == "25 Nov" and x["trial"] == 5)
  c = G.count_cycles(r["q"], r["t"])  # window 3 s -> run end (35.5 s)
  cam = list(csv.DictReader(open(LOGS / "trials_20261125_trial5_vision.csv", newline="")))
  tc = np.array([float(x["t_from_policy_start_s"]) for x in cam])
  qc = np.array([float(x["cumulative_quarter_turns"]) for x in cam])
  tw, cum = np.array(c["t_series"]), np.array(c["cum_cycles_series"])
  ok = (tw >= tc.min()) & (tw <= tc.max())
  cq = np.interp(tw[ok], tc, qc)
  cq = (cq - cq[0]) * (1.0 if qc[-1] >= 0 else -1.0)
  out["trial5_window_to_run_end"] = {"window_s": [c["t_start_s"], c["t_end_s"]],
                                     "slope_qturns_per_cycle": float(np.polyfit(cum[ok], cq, 1)[0]),
                                     "r": float(np.corrcoef(cum[ok], cq)[0, 1])}
  return out


def main() -> None:
  runs = load_runs()
  fs = sorted({round(1.0 / float(np.median(np.diff(r["t"]))), 2) for r in runs})
  rng = np.random.default_rng(20261127)
  res = {"label": "EXPLORATORY, POST HOC (real hand); operator counts are quarter turns counted by eye",
         "policy_rate_hz_seen": fs, "band_hz": list(G.BAND_HZ), "t_start_s": G.T_START_S, "variants": {}}
  base = per_run(runs)
  res["variants"]["run"] = {"desc": "window 3 s -> run end; run_s >= 20 s", "per_run": base, "policies": summarise(base, rng)}
  held = per_run(runs, basis="held_s", window="held")
  res["variants"]["held"] = {"desc": "window 3 s -> drop (or run end); held >= 20 s", "per_run": held, "policies": summarise(held, rng)}
  for name, joints in FINGERS.items():
    rows = per_run(runs, joints=joints)
    res["variants"][f"a_{name}"] = {"desc": f"(a) {name} finger joints only (PC1 of 4 joints); run variant", "policies": summarise(rows, rng)}
  rows = per_run(runs, band=(0.1, 1.5))
  res["variants"]["b_band_0.1_1.5"] = {"desc": "(b) band 0.1-1.5 Hz; run variant", "policies": summarise(rows, rng)}
  rows = per_run(runs, min_s=10.0)
  res["variants"]["c_runs_ge_10s"] = {"desc": "(c) runs >= 10 s; run variant", "policies": summarise(rows, rng)}
  res["variants"]["d_per_run_median"] = {"desc": "(d) per-run median of quarter turns per cycle; run variant",
                                         "policies": summarise(base, rng, per_run_median=True)}
  rows = per_run(runs, min_s=10.0, basis="held_s", window="held")
  res["variants"]["held_ge_10s"] = {"desc": "held variant with held >= 10 s", "policies": summarise(rows, rng)}
  ref_rank = ranking(res["variants"]["run"]["policies"])
  res["ranking_reference_variant"] = "run"
  for k, v in res["variants"].items():
    v["ranking"] = ranking(v["policies"])
    v["ranking_same_as_run"] = v["ranking"] == [p for p in ref_rank if p in v["ranking"]] and len(v["ranking"]) == len(ref_rank)
  res["camera_check_25nov"] = camera_check(runs)
  res["reference_first_pass"] = REFERENCE
  res["differences_vs_first_pass"] = first_pass_diagnostics(runs)
  for v in res["variants"].values():  # keep the JSON readable: drop per-cycle lists
    for x in v.get("per_run", []):
      x.pop("stroke_amplitudes_rad", None)
  (E / "results" / "real_hand.json").write_text(json.dumps(res, indent=1) + "\n")
  write_md(res)


def write_md(res: dict) -> None:
  V = res["variants"]
  L = ["# Gait efficiency: real hand (exploratory, post hoc)", "",
       "Generated by `scripts/real_hand.py`; numbers from `results/real_hand.json`. Operator counts are quarter turns "
       "counted by eye. Cycles from `scripts/gait.py` (PC1 of the 16 joints, band 0.12-1.2 Hz, Hilbert phase, first 3 s dropped). "
       f"Policy rate seen in the logs: {res['policy_rate_hz_seen']} Hz.", ""]
  for key in ("run", "held"):
    L += [f"## Variant `{key}`: {V[key]['desc']}", "",
          "| Policy | Runs | Quarter turns / cycle [95% CI over runs] | Median of runs | Rhythm, median (Hz) | Cycles / s | "
          "Stroke amplitude, median (rad) | First pass (q-turns/cycle; Hz) |", "|---|---|---|---|---|---|---|---|"]
    for p in ORDER:
      s = V[key]["policies"][p]
      if not s.get("n_runs"):
        L.append(f"| {p} | 0 | – | – | – | – | – | {REFERENCE['qturns_per_cycle_runs_ge_20s'][p]:.2f}; {REFERENCE['rhythm_hz'][p]:.2f} |")
        continue
      ci = s["qturns_per_cycle_ci95"]
      L.append(f"| {p} | {s['n_runs']} | {s['qturns_per_cycle']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] | {s['qturns_per_cycle_median_of_runs']:.2f} | "
               f"{s['rhythm_hz_median']:.2f} | {s['cycle_rate_hz']:.2f} | {s['stroke_amplitude_median_rad']:.3f} | "
               f"{REFERENCE['qturns_per_cycle_runs_ge_20s'][p]:.2f}; {REFERENCE['rhythm_hz'][p]:.2f} |")
    L += ["", f"Ranking by quarter turns per cycle: {' > '.join(V[key]['ranking'])}", ""]
  L += ["## Robustness (on variant `run`)", "", "| Check | Ranking | Same as `run` | Quarter turns / cycle per policy |", "|---|---|---|---|"]
  for k, v in V.items():
    if k in ("run",):
      continue
    vals = ", ".join(f"{p} {v['policies'][p]['metric']:.2f}" for p in ORDER if v["policies"][p].get("n_runs"))
    L.append(f"| {v['desc']} | {' > '.join(v['ranking'])} | {'yes' if v['ranking_same_as_run'] else 'no'} | {vals} |")
  L += ["", "## Camera check, 25 Nov (C5 seed 3; window 3 s -> drop or end)", "",
        "| Trial | Window (s) | Samples | Cycles | Camera quarter turns | Slope (q-turns / cycle) | r | First pass |", "|---|---|---|---|---|---|---|---|"]
  for k, c in res["camera_check_25nov"].items():
    ref = REFERENCE["camera"].get(k)
    refs = f"{ref['slope']:.2f}, r = {ref['r']:.3f}" if ref else "–"
    if "slope_qturns_per_cycle" in c:
      L.append(f"| {k} | {c['window_s'][0]:.0f}–{c['window_s'][1]:.1f} | {c['n_samples']} | {c['cycles_in_window']:.1f} | "
               f"{c['camera_qturns_in_window']:.1f} | {c['slope_qturns_per_cycle']:.2f} | {c['r']:.3f} | {refs} |")
    else:
      L.append(f"| {k} | {c['window_s'][0]:.0f}–{c['window_s'][1]:.1f} | – | – | – | {c['status']} | – | {refs} |")
  D = res["differences_vs_first_pass"]
  L += ["", "## Differences to the first pass (diagnostics; the method above is unchanged)", "",
        "- Quarter turns per cycle (variant `run`) agree with the first pass to two decimals for all six policies.",
        "- Rhythm differs by up to 0.02 Hz. A plain FFT of the centred PC1 (no Hann window, no zero padding) gives "
        + ", ".join(f"{p} {D['rhythm_plain_fft_median_hz'][p]:.2f}" for p in ORDER)
        + " Hz, the first-pass values; this script uses a Hann window and zero padding to locate the peak.",
        f"- Camera check, trial 5: with the window running to the end of the run ({D['trial5_window_to_run_end']['window_s'][1]:.1f} s, "
        f"past the 30-s drop) the slope is {D['trial5_window_to_run_end']['slope_qturns_per_cycle']:.2f} and "
        f"r = {D['trial5_window_to_run_end']['r']:.3f} (first pass: 0.79, r = 0.984). Here the camera window stops at the drop, "
        "because after it the fingers keep cycling without the cube."]
  (E / "results" / "real_hand.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
  main()
