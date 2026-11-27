#!/usr/bin/env python3
"""Seed jury, Step 1 (PLAN.md section 1): replay every real C5s1/C5s3 run through all three C5 seeds with the
kit's own code, check the gate, and test whether the seeds disagree more just before a drop.

Kit code used, unchanged: LEAP_Hand_RL_Deploy_Kit/kit/policy_runtime.py (LeapPolicy, one instance per seed, own
normaliser) and kit/deploy_direct.py (_hw_to_server, _server_to_hw, _HW_LIMIT_LO/HI, --limit-margin-rad default).
dynamixel_sdk is stubbed only so deploy_direct can be imported; no bus code runs.  CPU only, no hand.

    python training_source/project/experiments/seed_jury/scripts/real_replay.py
Writes results/real_replay.json and results/real_replay.md (stops after the gate if it fails).
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np

E = Path(__file__).resolve().parents[1]
ROOT = E.parents[3]
KIT = ROOT / "LEAP_Hand_RL_Deploy_Kit"
LOGS = KIT / "logs"

sys.path.insert(0, str(KIT / "kit"))
if "dynamixel_sdk" not in sys.modules:  # import-only stub; deploy_direct's bus classes are never used here
  _stub = types.ModuleType("dynamixel_sdk")
  _stub.__getattr__ = lambda name: object  # type: ignore[attr-defined]
  sys.modules["dynamixel_sdk"] = _stub
import deploy_direct as DD  # noqa: E402
from policy_runtime import LeapPolicy  # noqa: E402

SEEDS = {"C5s1": "C5", "C5s2": "C5s2", "C5s3": "C5s3"}  # name -> kit folder
EXPECTED_SHA = {"C5s1": "2a932dfe37a5540d", "C5s3": "c221af89199cdeab"}
SESSIONS = (("20 Nov", "trials.csv"), ("24 Nov", "trials_20261124.csv"), ("25 Nov", "trials_20261125.csv"))
LABEL = {"C5": "C5s1", "C5s3": "C5s3"}
FINGERS = ("index", "middle", "ring", "thumb")  # 4 contiguous hardware-order joints each
GATE_TOL = 1e-4
TIE_TOL = 1e-4  # rad; 'exceeds p99' needs value > p99 + TIE_TOL (deviation: ties at the action bound, see real_replay.md)
STEP_BOUND = 2.0 / 24.0  # per-finger ||4 x 1/24 rad|| = the largest possible executed step
SMOOTH = 10  # steps = 0.5 s at 20 Hz
NBOOT = 10000
FIG_RUNS = (("24 Nov", 15), ("25 Nov", 5), ("20 Nov", 12), ("24 Nov", 4))


def sha(p: Path) -> str:
  return hashlib.sha256(p.read_bytes()).hexdigest()


def margin() -> float:
  """deploy_direct's --limit-margin-rad default, read from its source (hand.env does not override it)."""
  import re
  m = re.search(r'"--limit-margin-rad",\s*type=float,\s*default=([0-9.]+)', Path(DD.__file__).read_text())
  return float(m.group(1))


def load_seeds() -> dict:
  out = {}
  for name, folder in SEEDS.items():
    p = LeapPolicy(KIT / "policies" / folder)
    out[name] = {"policy": p, "sha256": sha(Path(p.checkpoint_path)), "checkpoint": str(Path(p.checkpoint_path).relative_to(ROOT))}
  return out


def load_runs() -> list[dict]:
  runs = []
  for session, fname in SESSIONS:
    for r in csv.DictReader(open(LOGS / fname, newline="")):
      if r["policy"] not in LABEL:
        continue
      if r["intervened"].strip() == "True":
        continue
      dropped = r["dropped"].strip() == "True"
      runs.append({"session": session, "trial": int(r["trial"]), "seed": LABEL[r["policy"]], "log": r["log"],
                   "csv_sha256": r["checkpoint_sha256"], "dropped": dropped,
                   "t_drop": float(r["drop_time_s"]) if dropped else None, "run_s": float(r["run_s"]),
                   "stop_reason": r["stop_reason"], "notes": r["notes"]})
  return runs


def delta(a: np.ndarray, spec) -> np.ndarray:
  return spec.delta_min + 0.5 * (np.tanh(a) + 1.0) * (spec.delta_max - spec.delta_min)


def replay(run: dict, seeds: dict, lower: np.ndarray, upper: np.ndarray) -> dict:
  d = np.load(LOGS / run["log"], allow_pickle=False)
  assert int(np.max(d["stall_recovery_count"])) == 0, run["log"]
  q_hw, cmd_hw, raw_log = d["q_hw"], d["command_hw"], d["raw_action_server"]
  el = np.asarray(d["elapsed_s"], dtype=np.float64)
  T = len(el)
  ex = seeds[run["seed"]]["policy"]
  # reset pose (not logged): recovered from the step-0 command and raw action (PLAN 1.1)
  cmd0 = DD._hw_to_server(cmd_hw[0])
  q_reset = (cmd0 - delta(raw_log[0], ex.spec)).astype(np.float32)
  lo_s, hi_s = DD._hw_to_server(lower), DD._hw_to_server(upper)
  at_lim = ((np.abs(cmd0 - ex.lower_limits) < 1e-6) | (np.abs(cmd0 - ex.upper_limits) < 1e-6)
            | (np.abs(cmd0 - lo_s) < 1e-6) | (np.abs(cmd0 - hi_s) < 1e-6))
  q_reset[at_lim] = DD._hw_to_server(q_hw[0])[at_lim]
  for s in seeds.values():
    s["policy"].reset(q_reset)
  C = np.zeros((3, T, 16), np.float32)  # hardware order
  raw_ex = np.zeros((T, 16), np.float32)
  names = list(seeds)
  for k in range(T):
    q = DD._hw_to_server(q_hw[k])
    applied = q_reset if k == 0 else DD._hw_to_server(cmd_hw[k - 1])
    for i, n in enumerate(names):
      target, raw = seeds[n]["policy"].step(q, None, applied)
      C[i, k] = np.clip(DD._server_to_hw(target), lower, upper)
      if n == run["seed"]:
        raw_ex[k] = raw
  e = names.index(run["seed"])
  med = np.median(C, axis=0)
  diff = C[e] - med
  D = np.linalg.norm(diff, axis=1)
  Df = np.stack([np.linalg.norm(diff[:, 4 * f:4 * f + 4], axis=1) for f in range(4)], axis=1)
  step = np.zeros((T, 4))
  dc = np.diff(cmd_hw.astype(np.float64), axis=0)
  step[1:] = np.stack([np.linalg.norm(dc[:, 4 * f:4 * f + 4], axis=1) for f in range(4)], axis=1)
  return {"t": el - el[0], "D": D, "Df": Df, "Dbar": trailing(D), "Dbar_f": np.stack([trailing(Df[:, f]) for f in range(4)], 1),
          "cmd_step_f": step, "gate_raw_max_err": float(np.max(np.abs(raw_ex - raw_log))),
          "gate_cmd_max_err": float(np.max(np.abs(C[e] - cmd_hw))), "n_steps": T, "n_reset_joints_from_q": int(at_lim.sum())}


def trailing(x: np.ndarray, n: int = SMOOTH) -> np.ndarray:
  c = np.cumsum(np.insert(np.asarray(x, np.float64), 0, 0.0))
  k = np.arange(1, len(x) + 1)
  lo = np.maximum(0, k - n)
  return (c[k] - c[lo]) / (k - lo)


def wmask(t: np.ndarray, a: float, b: float) -> np.ndarray:
  return (t >= a) & (t < b)


def score(rep: dict, a: float, b: float, key: str = "Dbar", f: int | None = None) -> float | None:
  m = wmask(rep["t"], a, b)
  if m.sum() == 0:
    return None
  x = rep[key] if f is None else rep[key][:, f]
  return float(x[m].mean())


def control_windows(rep: dict, t_end: float) -> list[tuple[float, float]]:
  out, a = [], 3.0
  while a + 2.0 <= t_end + 1e-9:
    out.append((a, a + 2.0))
    a += 2.0
  return out


def t_end_of(run: dict, rep: dict, t_drop: float | None) -> float:
  return (t_drop - 5.0) if t_drop is not None else float(rep["t"][-1])


def auc(pos: list[float], neg: list[float]) -> float:
  p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
  return float(((p > n).sum() + 0.5 * (p == n).sum()) / (p.size * n.size))


def boot_median(x: list[float], rng) -> list[float]:
  x = np.asarray(x)
  b = np.median(x[rng.integers(0, len(x), (NBOOT, len(x)))], axis=1)
  return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def h_tests(P: list[dict], H: list[dict], win: tuple[float, float], rng, drop_key: str = "t_drop") -> dict:
  """H1 and H2 for one pre-drop window definition (offsets before the drop)."""
  per = []
  for r in P:
    td = r[drop_key]
    pre = score(r["rep"], td - win[0], td - win[1])
    ctl = [score(r["rep"], a, b) for a, b in control_windows(r, t_end_of(r, r["rep"], td))]
    per.append({"run": f"{r['session']} t{r['trial']}", "seed": r["seed"], "t_drop": td, "pre": pre, "n_control": len(ctl),
                "control_median": float(np.median(ctl)), "ratio": pre / float(np.median(ctl)), "control": ctl})
  ratios = [x["ratio"] for x in per]
  h1 = {"median_ratio": float(np.median(ratios)), "ci95": boot_median(ratios, rng), "n_runs": len(ratios),
        "n_ratio_gt_1": int(sum(x > 1 for x in ratios)),
        "per_seed_median": {s: float(np.median([x["ratio"] for x in per if x["seed"] == s])) for s in ("C5s1", "C5s3")}}
  h1["supported"] = bool(h1["median_ratio"] > 1 and h1["ci95"][0] > 1)
  pos = [x["pre"] for x in per]
  neg = [c for x in per for c in x["control"]]
  a0 = auc(pos, neg)
  bs = np.empty(NBOOT)
  for i in range(NBOOT):
    j = rng.integers(0, len(per), len(per))
    bs[i] = auc([per[k]["pre"] for k in j], [c for k in j for c in per[k]["control"]])
  h2 = {"auc": a0, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n_pos": len(pos), "n_neg": len(neg)}
  h2["supported"] = bool(h2["ci95"][0] > 0.5)
  # secondary: negatives also include the Held runs' control windows (stratified cluster bootstrap)
  hneg = [[score(r["rep"], a, b) for a, b in control_windows(r, t_end_of(r, r["rep"], None))] for r in H]
  a1 = auc(pos, neg + [c for h in hneg for c in h])
  bs1 = np.empty(NBOOT)
  for i in range(NBOOT):
    j = rng.integers(0, len(per), len(per))
    jh = rng.integers(0, len(H), len(H))
    bs1[i] = auc([per[k]["pre"] for k in j], [c for k in j for c in per[k]["control"]] + [c for k in jh for c in hneg[k]])
  h2["secondary_with_held"] = {"auc": a1, "ci95": [float(np.percentile(bs1, 2.5)), float(np.percentile(bs1, 97.5))],
                               "n_neg": len(neg) + sum(len(h) for h in hneg)}
  for x in per:
    x.pop("control")
  return {"window_before_drop_s": list(win), "per_run": per, "H1": h1, "H2": h2}


def camera_drop_time(trial: int) -> float:
  rows = list(csv.DictReader(open(LOGS / f"trials_20261125_trial{trial}_vision.csv", newline="")))
  t = np.array([float(x["t_from_policy_start_s"]) for x in rows])
  q = np.array([float(x["cumulative_quarter_turns"]) for x in rows])
  far = np.flatnonzero(np.abs(q - q[-1]) > 0.25)
  return float(t[0] if far.size == 0 else t[min(far[-1] + 1, len(t) - 1)])


def main() -> None:
  rng = np.random.default_rng(20261128)
  seeds = load_seeds()
  lower, upper = DD._HW_LIMIT_LO + margin(), DD._HW_LIMIT_HI - margin()
  runs = load_runs()
  for r in runs:
    assert r["csv_sha256"].startswith(EXPECTED_SHA[r["seed"]]) and seeds[r["seed"]]["sha256"] == r["csv_sha256"], r
    r["rep"] = replay(r, seeds, lower, upper)
  gate = {"tolerance": GATE_TOL, "raw_max_err": max(r["rep"]["gate_raw_max_err"] for r in runs),
          "cmd_max_err_rad": max(r["rep"]["gate_cmd_max_err"] for r in runs), "n_runs": len(runs),
          "n_steps": int(sum(r["rep"]["n_steps"] for r in runs)),
          "per_run": {f"{r['session']} t{r['trial']}": [r["rep"]["gate_raw_max_err"], r["rep"]["gate_cmd_max_err"]] for r in runs},
          "reset_joints_taken_from_q_hw0": int(sum(r["rep"]["n_reset_joints_from_q"] for r in runs))}
  gate["passed"] = bool(gate["raw_max_err"] <= GATE_TOL)
  out = {"plan": "PLAN.md (sha256 in logs/plan.sha256)", "label": "PRE-REGISTERED analysis of already-collected hardware logs; small n",
         "seeds": {k: {kk: v[kk] for kk in ("sha256", "checkpoint")} for k, v in seeds.items()},
         "limits_margin_rad": margin(), "gate": gate}
  if not gate["passed"]:
    (E / "results" / "real_replay.json").write_text(json.dumps(out, indent=1) + "\n")
    print("GATE FAILED", json.dumps(gate, indent=1))
    sys.exit(2)

  P = [r for r in runs if r["dropped"] and r["t_drop"] >= 10]
  EARLY = [r for r in runs if r["dropped"] and r["t_drop"] < 10]
  H = [r for r in runs if not r["dropped"]]
  out["sets"] = {k: [f"{r['session']} t{r['trial']} ({r['seed']})" for r in v] for k, v in (("P", P), ("Early", EARLY), ("Held", H))}
  out["primary"] = h_tests(P, H, (3.0, 1.0), rng)
  out["sensitivity"] = h_tests(P, H, (2.0, 0.5), rng)

  # per finger H1 (descriptive)
  pf = {}
  for f, name in enumerate(FINGERS):
    rs = []
    for r in P:
      pre = score(r["rep"], r["t_drop"] - 3, r["t_drop"] - 1, "Dbar_f", f)
      ctl = [score(r["rep"], a, b, "Dbar_f", f) for a, b in control_windows(r, r["t_drop"] - 5)]
      rs.append(pre / float(np.median(ctl)))
    pf[name] = {"median_ratio": float(np.median(rs)), "ci95": boot_median(rs, rng), "ratios": rs}
  out["per_finger_H1"] = pf

  # early drops vs the same seed's Held runs
  early = []
  for r in EARLY:
    a, b = max(0.5, r["t_drop"] - 3), r["t_drop"] - 1
    if b - a < 0.5:
      early.append({"run": f"{r['session']} t{r['trial']}", "seed": r["seed"], "t_drop": r["t_drop"], "status": "window < 0.5 s"})
      continue
    s = score(r["rep"], a, b)
    hs = [h for h in H if h["seed"] == r["seed"]]
    tm = [score(h["rep"], a, b) for h in hs]
    pooled = [score(h["rep"], x, y) for h in hs for x, y in control_windows(h, t_end_of(h, h["rep"], None))]
    early.append({"run": f"{r['session']} t{r['trial']}", "seed": r["seed"], "t_drop": r["t_drop"], "window": [a, b], "score": s,
                  "time_matched_held": tm, "time_matched_percentile": float(100 * np.mean(np.asarray(tm) < s)),
                  "ratio_time_matched_median": s / float(np.median(tm)),
                  "ratio_pooled_control_median": s / float(np.median(pooled)), "notes": r["notes"]})
  ok = [x for x in early if "score" in x]
  out["early_drops"] = {"per_run": early, "summary": {
      "n": len(ok), "median_ratio_time_matched": float(np.median([x["ratio_time_matched_median"] for x in ok])),
      "median_ratio_pooled_control": float(np.median([x["ratio_pooled_control_median"] for x in ok])),
      "n_above_all_time_matched": int(sum(x["time_matched_percentile"] == 100.0 for x in ok))}}

  # largest single-finger outlier move per pre-drop window vs p99 of control steps (same seed, P + Held runs)
  p99 = {}
  for sd in ("C5s1", "C5s3"):
    dd, cs = [], []
    for r in [x for x in P + H if x["seed"] == sd]:
      m = np.zeros_like(r["rep"]["t"], dtype=bool)
      for a, b in control_windows(r, t_end_of(r, r["rep"], r["t_drop"])):
        m |= wmask(r["rep"]["t"], a, b)
      dd.append(r["rep"]["Df"][m])
      cs.append(r["rep"]["cmd_step_f"][m])
    dd, cs = np.concatenate(dd), np.concatenate(cs)
    p99[sd] = {"Df": np.percentile(dd, 99, axis=0).tolist(), "cmd_step": np.percentile(cs, 99, axis=0).tolist()}
  outl = []
  for r in P + EARLY:
    a, b = max(0.5, r["t_drop"] - 3), r["t_drop"] - 1
    m = wmask(r["rep"]["t"], a, b)
    if m.sum() == 0:
      continue
    idx = np.flatnonzero(m)
    Df, cs = r["rep"]["Df"][idx], r["rep"]["cmd_step_f"][idx]
    i, f = np.unravel_index(np.argmax(Df), Df.shape)
    i2, f2 = np.unravel_index(np.argmax(cs), cs.shape)
    outl.append({"run": f"{r['session']} t{r['trial']}", "seed": r["seed"], "set": "P" if r in P else "Early", "t_drop": r["t_drop"],
                 "disagreement": {"finger": FINGERS[f], "t": float(r["rep"]["t"][idx[i]]), "value": float(Df[i, f]),
                                  "p99_control": p99[r["seed"]]["Df"][f], "exceeds_p99": bool(Df[i, f] > p99[r["seed"]]["Df"][f] + TIE_TOL)},
                 "cmd_step": {"finger": FINGERS[f2], "t": float(r["rep"]["t"][idx[i2]]), "value": float(cs[i2, f2]),
                              "p99_control": p99[r["seed"]]["cmd_step"][f2], "exceeds_p99": bool(cs[i2, f2] > p99[r["seed"]]["cmd_step"][f2] + TIE_TOL),
                              "at_action_bound": bool(cs[i2, f2] >= STEP_BOUND - TIE_TOL)},
                 "notes": r["notes"]})
  out["outlier_moves"] = {"p99_control_per_finger": p99, "per_window": outl, "tie_tol_rad": TIE_TOL, "step_bound_rad": STEP_BOUND,
                          "summary": {s: {"n": sum(1 for x in outl if x["set"] == s),
                                          "disagreement_exceeds_p99": sum(x["disagreement"]["exceeds_p99"] for x in outl if x["set"] == s),
                                          "cmd_step_exceeds_p99": sum(x["cmd_step"]["exceeds_p99"] for x in outl if x["set"] == s),
                                          "cmd_step_at_bound": sum(x["cmd_step"]["at_action_bound"] for x in outl if x["set"] == s),
                                          "disagreement_finger_counts": {fn: sum(x["disagreement"]["finger"] == fn for x in outl if x["set"] == s) for fn in FINGERS}}
                                      for s in ("P", "Early")}}

  # 25 Nov camera check
  cam = {}
  for tr in (5, 4):
    r = next(x for x in runs if x["session"] == "25 Nov" and x["trial"] == tr)
    tc = camera_drop_time(tr)
    cam[f"trial{tr}"] = {"operator_drop_s": r["t_drop"], "camera_drop_s": tc}
    if tr == 5:
      ctl = [score(r["rep"], a, b) for a, b in control_windows(r, tc - 5)]
      cam["trial5"]["ratio_camera"] = score(r["rep"], tc - 3, tc - 1) / float(np.median(ctl))
      rat = [x["ratio"] for x in out["primary"]["per_run"] if x["run"] != "25 Nov t5"] + [cam["trial5"]["ratio_camera"]]
      cam["trial5"]["H1_median_ratio_with_camera"] = float(np.median(rat))
      cam["trial5"]["ratio_operator"] = next(x["ratio"] for x in out["primary"]["per_run"] if x["run"] == "25 Nov t5")
    else:
      a, b = max(0.5, tc - 3), tc - 1
      if b - a >= 0.5:
        s = score(r["rep"], a, b)
        tm = [score(h["rep"], a, b) for h in H if h["seed"] == "C5s3"]
        cam["trial4"].update(score=s, ratio_time_matched_median=s / float(np.median(tm)))
      else:
        cam["trial4"]["status"] = "camera pre-drop window < 0.5 s"
  out["camera_check_25nov"] = cam

  # real C5s3 control-window steps, for the simulation comparison (PLAN 2.3)
  vals = []
  for r in [x for x in P + H if x["seed"] == "C5s3"]:
    m = np.zeros_like(r["rep"]["t"], dtype=bool)
    for a, b in control_windows(r, t_end_of(r, r["rep"], r["t_drop"])):
      m |= wmask(r["rep"]["t"], a, b)
    vals.append(r["rep"]["Dbar"][m])
  v = np.concatenate(vals)
  out["real_C5s3_control_Dbar"] = {"n_steps": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)), "p95": float(np.percentile(v, 95))}
  (E / "results").mkdir(exist_ok=True)
  (E / "results" / "real_replay.json").write_text(json.dumps(out, indent=1) + "\n")
  write_md(out)
  print((E / "results" / "real_replay.md").read_text())


def f(x, d=3):
  return "–" if x is None else f"{x:.{d}f}"


def write_md(o: dict) -> None:
  g, p, s = o["gate"], o["primary"], o["sensitivity"]
  L = ["# Seed jury, Step 1: real-hand replay through the three C5 seeds", "",
       f"Generated by `scripts/real_replay.py`. **{o['label']}.** Plan: `PLAN.md` (hashed before any value was computed).", "",
       "## Gate", "",
       f"- Runs replayed: {g['n_runs']} ({g['n_steps']} steps), with the kit's `policy_runtime.LeapPolicy` and `deploy_direct` joint maps and limits.",
       f"- Max |replayed − logged raw action| = **{g['raw_max_err']:.2e}** (tolerance {g['tolerance']:.0e}): **{'PASSED' if g['passed'] else 'FAILED'}**.",
       f"- Max |replayed − logged executed command| = {g['cmd_max_err_rad']:.2e} rad.",
       f"- Reset joints taken from q_hw[0] (step-0 command at a limit): {g['reset_joints_taken_from_q_hw0']}.",
       f"- Seed checkpoints: " + "; ".join(f"{k} `{v['sha256'][:16]}`" for k, v in o["seeds"].items()) + ".", "",
       "## H1 and H2 (primary set P: drops at ≥ 10 s, n = 8)", "",
       "| Window before the drop | H1 median ratio [95% CI] | runs with ratio > 1 | H1 | H2 AUC [95% CI] | H2 | AUC incl. held-run controls [95% CI] |",
       "|---|---|---|---|---|---|---|"]
  for name, x in (("pre-drop 3 → 1 s (primary)", p), ("sensitivity 2 → 0.5 s", s)):
    h1, h2 = x["H1"], x["H2"]
    L.append(f"| {name} | {f(h1['median_ratio'],2)} [{f(h1['ci95'][0],2)}, {f(h1['ci95'][1],2)}] | {h1['n_ratio_gt_1']} of {h1['n_runs']} | "
             f"{'supported' if h1['supported'] else 'not supported'} | {f(h2['auc'],2)} [{f(h2['ci95'][0],2)}, {f(h2['ci95'][1],2)}] | "
             f"{'supported' if h2['supported'] else 'not supported'} | {f(h2['secondary_with_held']['auc'],2)} "
             f"[{f(h2['secondary_with_held']['ci95'][0],2)}, {f(h2['secondary_with_held']['ci95'][1],2)}] |")
  L += ["", "Per run (primary window; D̄ in rad, trailing 0.5-s mean of ‖executing seed − per-joint median of the three‖):", "",
        "| Run | Seed | Drop s | Pre-drop D̄ | Control median D̄ (n windows) | Ratio |", "|---|---|---|---|---|---|"]
  for x in p["per_run"]:
    L.append(f"| {x['run']} | {x['seed']} | {x['t_drop']:g} | {f(x['pre'],4)} | {f(x['control_median'],4)} ({x['n_control']}) | {f(x['ratio'],2)} |")
  L += ["", f"Per seed median ratio: C5s1 {f(p['H1']['per_seed_median']['C5s1'],2)} (2 runs), C5s3 {f(p['H1']['per_seed_median']['C5s3'],2)} (6 runs).", "",
        "## Per finger (descriptive; H1 ratio from the per-finger D̄)", "", "| Finger | Median ratio [95% CI] |", "|---|---|"]
  for k, v in o["per_finger_H1"].items():
    L.append(f"| {k} | {f(v['median_ratio'],2)} [{f(v['ci95'][0],2)}, {f(v['ci95'][1],2)}] |")
  es = o["early_drops"]["summary"]
  L += ["", "## Early drops (< 10 s; descriptive)", "",
        f"{es['n']} early drops. Median ratio to the same seed's held runs, same absolute window: **{f(es['median_ratio_time_matched'],2)}**; "
        f"to the held runs' pooled control windows: {f(es['median_ratio_pooled_control'],2)}. Above every time-matched held value: {es['n_above_all_time_matched']} of {es['n']}.", "",
        "| Run | Seed | Drop s | Window s | D̄ | Ratio (time-matched) | Percentile among held | Ratio (pooled control) | Notes |", "|---|---|---|---|---|---|---|---|---|"]
  for x in o["early_drops"]["per_run"]:
    if "score" not in x:
      L.append(f"| {x['run']} | {x['seed']} | {x['t_drop']:g} | – | – | – | – | – | {x['status']} |")
      continue
    L.append(f"| {x['run']} | {x['seed']} | {x['t_drop']:g} | {x['window'][0]:g}–{x['window'][1]:g} | {f(x['score'],4)} | {f(x['ratio_time_matched_median'],2)} | "
             f"{x['time_matched_percentile']:.0f} | {f(x['ratio_pooled_control_median'],2)} | {x['notes'][:60]} |")
  om = o["outlier_moves"]
  L += ["", "## Largest single-finger move in each pre-drop window (descriptive)", "",
        "Disagreement = unsmoothed per-finger ‖executing − median‖; command step = executed per-finger ‖c_k − c_k−1‖; each against the 99th percentile of the same seed's control-window steps.",
        "",
        f"**Deviation from the plan (tie handling, not specified there):** the executed per-finger step is often at the action bound, {o['outlier_moves']['step_bound_rad']:.4f} rad (all 4 joints at 1/24 rad), "
        f"in control windows too, so p99 sits at that bound. 'Exceeds' therefore needs value > p99 + {o['outlier_moves']['tie_tol_rad']:.0e} rad; without this tolerance, float-rounding ties of < 1e-5 rad were counted as exceedances. ⊓ = at the bound.", "",
        "| Run | Set | Disagreement: finger, t, value / p99 | Command step: finger, t, value / p99 | Notes |", "|---|---|---|---|---|"]
  for x in om["per_window"]:
    dg, cs = x["disagreement"], x["cmd_step"]
    L.append(f"| {x['run']} ({x['seed']}) | {x['set']} | {dg['finger']}, {dg['t']:.1f} s, {dg['value']:.4f} / {dg['p99_control']:.4f}{' **>p99**' if dg['exceeds_p99'] else ''} | "
             f"{cs['finger']}, {cs['t']:.1f} s, {cs['value']:.4f} / {cs['p99_control']:.4f}{' ⊓' if cs['at_action_bound'] else ''}{' **>p99**' if cs['exceeds_p99'] else ''} | {x['notes'][:50]} |")
  for k, v in om["summary"].items():
    L.append("")
    L.append(f"- {k}: disagreement above p99 in {v['disagreement_exceeds_p99']} of {v['n']} windows; command step above p99 in {v['cmd_step_exceeds_p99']} of {v['n']} (at the action bound in {v['cmd_step_at_bound']}); "
             f"finger with the largest disagreement: " + ", ".join(f"{fn} {c}" for fn, c in v["disagreement_finger_counts"].items()) + ".")
  c = o["camera_check_25nov"]
  L += ["", "## 25 Nov camera check", "",
        f"- Trial 5: operator drop {c['trial5']['operator_drop_s']:g} s, camera drop {c['trial5']['camera_drop_s']:.1f} s; ratio {f(c['trial5']['ratio_operator'],2)} (operator) → "
        f"{f(c['trial5']['ratio_camera'],2)} (camera); H1 median with the camera time {f(c['trial5']['H1_median_ratio_with_camera'],2)}.",
        f"- Trial 4 (early): operator drop {c['trial4']['operator_drop_s']:g} s, camera drop {c['trial4']['camera_drop_s']:.1f} s; "
        + (f"time-matched ratio {f(c['trial4']['ratio_time_matched_median'],2)}." if "score" in c["trial4"] else c["trial4"].get("status", "")), "",
        "## Real C5s3 control-window D̄ (for the simulation comparison)", "",
        f"{o['real_C5s3_control_Dbar']['n_steps']} steps: mean {f(o['real_C5s3_control_Dbar']['mean'],4)}, median {f(o['real_C5s3_control_Dbar']['median'],4)}, "
        f"95th percentile {f(o['real_C5s3_control_Dbar']['p95'],4)} rad."]
  (E / "results" / "real_replay.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
  main()
