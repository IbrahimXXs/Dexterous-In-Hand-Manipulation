#!/usr/bin/env python3
"""Offline actuator / policy audit of real-hand run logs (no hardware access).

Reads every ``logs/physical_<date>_*.npz`` written by ``kit/deploy_direct.py`` plus the
operator record ``logs/trials.csv`` and produces, per attempt, per joint and per finger:
outcome classification, current, tracking error, effective response lag, joint/command
velocity, loop timing, guard-stop event reports, quarter-turn conversion, data-quality flags;
optionally the same metrics on simulation rollouts (``--sim-dir``) for a like-for-like check.

Signal semantics (established from deploy_direct.py, not from field names):
  * one record per policy step k; the state (q_hw, qd_hw, current_ma) is read at loop start,
    the guards are evaluated on that read, then the policy runs and ``command_hw[k]`` is
    written.  A read that trips a guard raises BEFORE its record is appended: the tripping
    sample is never in the log (the last logged sample is one control period earlier).
  * elapsed_s[k]      = monotonic seconds since the bridge started (before cube placement,
                        ramp ...) taken after the command write; loop_compute_s[k] = time from
                        loop start (state read) to that point.  t_read = elapsed - compute.
  * q_hw / command_hw = rad, hardware joint order; command_hw is the clipped goal position
                        actually written to the motors (after the runtime's +-1/24 rad step).
  * qd_hw             = Dynamixel present velocity (0.229 rpm units) in rad/s.
  * current_ma        = Dynamixel present current register (signed), as mA (kit convention).
  * temperature_c     = read every 20th step only; NaN otherwise (not missing data).
  * tracking_error_rad[k] = max_j |q_hw[k, j] - command_hw[k-1, j]| (scalar; at k = 0 the
                        previous command is the measured post-ramp pose, which is not logged).
  * stop_reason       = "max_steps" | "interrupted" (Ctrl+C) | "error: <guard message>";
                        logs written before the 20 Nov patch say only "error".

    python kit/audit_hardware_logs.py --date 20261120 --out audit/20261120 \
        [--sim-dir ../training_source/project/experiments/c5_hw_v1/sim_reference]

Never connects to the hand, never writes to logs/ (the raw data are read-only inputs).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

KIT = Path(__file__).resolve().parents[1]
# row 0 of the per-joint tracking error is NaN by construction (previous command not logged)
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

# --------------------------------------------------------------------------------------- hand
HW_JOINTS = ("if_rot", "if_mcp", "if_pip", "if_dip", "mf_rot", "mf_mcp", "mf_pip", "mf_dip",
             "rf_rot", "rf_mcp", "rf_pip", "rf_dip", "th_cmc", "th_axl", "th_mcp", "th_ipl")
FINGERS = {"index": (0, 1, 2, 3), "middle": (4, 5, 6, 7), "ring": (8, 9, 10, 11), "thumb": (12, 13, 14, 15)}
# server (training) order = hardware order with rot/mcp swapped for the three fingers (deploy_direct.py)
HW_TO_SERVER = np.array([1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15], dtype=np.int64)
SERVER_TO_HW = np.argsort(HW_TO_SERVER).astype(np.int64)
SERVER_JOINTS = tuple(HW_JOINTS[i] for i in HW_TO_SERVER)
HW_LIMIT_LO = np.array([-0.523599, -0.261799, -0.261799, -0.261799] * 3 + [0.0, 0.0, -0.349066, -0.349066])
HW_LIMIT_HI = np.array([0.523599, 1.91986, 1.91986, 1.91986] * 3 + [1.571, 1.571, 1.74533, 1.74533])
LIMIT_MARGIN_RAD = 0.01           # deploy_direct --limit-margin-rad default (not overridden by leap.sh)
COMMAND_DELTA_MAX = 1.0 / 24.0    # policy runtime: delta action in [-1/24, +1/24] rad per step
REQUIRED = ("elapsed_s", "loop_compute_s", "q_hw", "qd_hw", "current_ma", "temperature_c", "command_hw",
            "tracking_error_rad", "stop_reason", "completed")
PER_JOINT_FIELDS = ("q_hw", "qd_hw", "current_ma", "temperature_c", "command_hw")

# ---------------------------------------------------------------------------------- analysis
CURRENT_LEVELS_MA = (200.0, 300.0, 340.0)  # 340 mA = 97% of the 350 mA limit ("at the limit")
SAT_MA = 340.0
JAM_TRACKING_RAD = 0.2                     # saturated AND this far from the command = blocked
SUSTAINED_S = 0.5
WINDOWS_S = ((0, 2), (2, 4), (4, 8), (8, 16), (16, 30), (30, 61))
LAG_GRID_S = np.round(np.arange(0.0, 0.6001, 0.005), 6)
LAG_MAX_CURRENT_MA = 200.0                 # lag fitted only on unsaturated samples of that joint
LAG_MIN_SAMPLES = 100
QUASI_STATIC_RAD_S = 0.15
DROP_GUARD_MARGIN_S = 2.0                  # operator drop time vs automatic stop: order unresolved within this
JAM_WINDOW_S = 1.0                         # jam episode: >= JAM_MIN_FRAC of a 1 s window blocked on one joint
JAM_MIN_FRAC = 0.5
START_TRANSIENT_S = 3.0                    # trip within 3 s of policy start with the last logged |I| below SAT_MA
CATEGORIES = {1: "completed_60s", 2: "dropped", 3: "guard_stop", 4: "operator_stop_documented",
              5: "operator_stop_unknown_reason", 6: "invalid_or_incomplete"}


def _f(x: Any) -> float:
  try:
    return float(x)
  except (TypeError, ValueError):
    return float("nan")


def read_hand_env(path: Path = KIT / "configs" / "hand.env") -> dict[str, str]:
  """KEY=VALUE lines of configs/hand.env (``${VAR:-default}`` -> default)."""
  out: dict[str, str] = {}
  if not path.exists():
    return out
  for line in path.read_text().splitlines():
    m = re.match(r"^\s*([A-Z_][A-Z0-9_]*)=(\S+)", line)
    if m:
      v = m.group(2)
      d = re.match(r"^\$\{[A-Z_0-9]+:-([^}]*)\}$", v)
      out[m.group(1)] = d.group(1) if d else v
  return out


# ------------------------------------------------------------------------------- loading
@dataclass
class RunLog:
  path: Path
  name: str
  f: dict[str, np.ndarray | None]
  meta: dict[str, Any]
  stop_reason: str | None
  completed: bool | None
  started_at: str | None
  n: int
  flags: list[str] = field(default_factory=list)

  @property
  def ok(self) -> bool:
    return not any(fl.startswith(("missing:", "shape:", "no_samples")) for fl in self.flags)


def load_run(path: Path) -> RunLog:
  """Load one log; missing / malformed fields become None (flagged), never zeros."""
  flags: list[str] = []
  try:
    d = np.load(path, allow_pickle=False)
    files = set(d.files)
  except Exception as exc:  # unreadable / truncated file
    return RunLog(path, path.name, {k: None for k in REQUIRED}, {}, None, None, None, 0, [f"unreadable:{exc}"])
  f: dict[str, np.ndarray | None] = {}
  for k in REQUIRED:
    if k not in files:
      flags.append(f"missing:{k}")
      f[k] = None
    else:
      f[k] = np.asarray(d[k])
  n = int(f["elapsed_s"].shape[0]) if f["elapsed_s"] is not None and f["elapsed_s"].ndim == 1 else 0
  if n == 0:
    flags.append("no_samples")
  for k in PER_JOINT_FIELDS:
    a = f.get(k)
    if a is not None and a.shape != (n, 16):
      flags.append(f"shape:{k}={a.shape}")
      f[k] = None
  for k in ("elapsed_s", "loop_compute_s", "tracking_error_rad"):
    a = f.get(k)
    if a is not None and a.shape != (n,):
      flags.append(f"shape:{k}={a.shape}")
      f[k] = None
  for k in ("q_hw", "qd_hw", "current_ma", "command_hw", "loop_compute_s", "tracking_error_rad"):
    a = f.get(k)
    if a is not None:
      a = a.astype(np.float64)
      f[k] = a
      if not np.all(np.isfinite(a)):
        flags.append(f"nonfinite:{k}={int(np.sum(~np.isfinite(a)))}")
  t = f.get("elapsed_s")
  if t is not None:
    t = t.astype(np.float64)
    f["elapsed_s"] = t
    if not np.all(np.isfinite(t)):
      flags.append("timestamps_nonfinite")
    elif n > 1 and np.any(np.diff(t) <= 0):
      flags.append("timestamps_nonmonotonic")
  meta: dict[str, Any] = {}
  if "run_meta_json" in files:
    try:
      meta = json.loads(str(d["run_meta_json"]))
    except Exception:
      flags.append("meta_unparseable")
  else:
    flags.append("missing:run_meta_json")
  stop = str(f["stop_reason"]) if f.get("stop_reason") is not None else None
  comp = bool(f["completed"]) if f.get("completed") is not None else None
  started = str(d["started_at"]) if "started_at" in files else None
  if any(k.startswith("mon_") for k in files):
    flags.append("has_mon_fields_ignored")  # learned-estimator outputs: not used in this audit
  if "_part" in path.stem:
    flags.append("chunked_part_file")
  return RunLog(path, path.name, f, meta, stop, comp, started, n, flags)


def read_trials(csv_path: Path) -> list[dict[str, str]]:
  if not csv_path.exists():
    return []
  with open(csv_path, newline="") as fh:
    return list(csv.DictReader(fh))


def match_trials(runs: list[RunLog], rows: list[dict[str, str]]) -> tuple[dict[str, dict], list[str], list[str]]:
  """CSV row per log file name; returns (map, rows without a file, duplicate rows)."""
  by: dict[str, dict] = {}
  dup: list[str] = []
  for r in rows:
    if r.get("log") in by:
      dup.append(r["log"])
    by[r.get("log", "")] = r
  names = {r.name for r in runs}
  orphans = [r["log"] for r in rows if r.get("log") not in names]
  return {r.name: by[r.name] for r in runs if r.name in by}, orphans, dup


def identify_policy(run: RunLog, row: dict | None, sha_by_policy: dict[str, set[str]]) -> tuple[str, list[str]]:
  """Policy from run metadata (folder name + checkpoint sha256), cross-checked with the CSV."""
  notes: list[str] = []
  pd = run.meta.get("policy_dir")
  name = Path(str(pd)).name if pd else None
  sha = run.meta.get("policy_sha256")
  by_sha = [p for p, s in sha_by_policy.items() if sha and sha in s]
  if name is None and row is not None:
    name = row["policy"]
    notes.append("policy_from_csv_only")
  if name is None and len(by_sha) == 1:
    name = by_sha[0]
    notes.append("policy_from_sha_only")
  if name is None:
    return "unknown", notes + ["policy_unidentified"]
  if by_sha and name not in by_sha:
    notes.append(f"sha_mismatch:{by_sha}")
  if row is not None and row["policy"] != name:
    notes.append(f"csv_policy_mismatch:{row['policy']}")
  if sha and row is not None and row.get("checkpoint_sha256") and row["checkpoint_sha256"] != sha:
    notes.append("csv_sha_mismatch")
  return name, notes


# ---------------------------------------------------------------------------- classification
def guard_info(stop: str | None) -> tuple[str | None, float]:
  """(guard type, trip value) from the stop reason; ('unrecorded', nan) for a bare 'error'."""
  if stop is None or not stop.startswith("error"):
    return None, float("nan")
  if stop.strip() == "error":
    return "unrecorded", float("nan")
  m = re.search(r"(Current|Velocity|Temperature|Tracking-error|Command-step|Ramp current|Ramp velocity) guard tripped:?\s*(.*)", stop)
  if not m:
    return "other_error", float("nan")
  num = re.search(r"([0-9]+\.?[0-9]*)\s*(mA|rad/s|C|rad)", m.group(2))
  return m.group(1).lower().replace(" ", "_").replace("-", "_"), float(num.group(1)) if num else float("nan")


def classify(run: RunLog, row: dict | None, tracking_abort_rad: float = 0.65) -> dict[str, Any]:
  """One mutually exclusive outcome category per attempt; ambiguities are flagged."""
  flags: list[str] = []
  dur = duration_s(run)
  dropped = row is not None and row.get("dropped") == "True"
  drop_t = _f(row.get("drop_time_s")) if row is not None else float("nan")
  gtype, gval = guard_info(run.stop_reason)
  inferred = None
  if not run.ok or run.stop_reason is None:
    return {"category": 6, "category_name": CATEGORIES[6], "confidence": "high", "guard_type": gtype,
            "guard_value": gval, "flags": ["missing_or_malformed_fields"], "reason": "required fields missing/malformed"}
  if run.stop_reason == "running":
    return {"category": 6, "category_name": CATEGORIES[6], "confidence": "high", "guard_type": None,
            "guard_value": float("nan"), "flags": ["chunk_still_running"], "reason": "chunk file of an unfinished run"}
  if gtype is not None:
    if gtype == "unrecorded":
      flags.append("stop_message_not_logged_pre_patch")
      te = run.f["tracking_error_rad"]
      if te is not None and len(te) and float(te[-1]) >= 0.9 * tracking_abort_rad:
        inferred = "tracking_error (inferred: last logged max tracking error "
        inferred += f"{float(te[-1]):.3f} rad vs abort {tracking_abort_rad} rad)"
      else:
        inferred = "unknown"
    if dropped and math.isfinite(drop_t):
      if drop_t < dur - DROP_GUARD_MARGIN_S:
        flags.append(f"guard_after_operator_drop(drop {drop_t:g} s, stop {dur:.1f} s)")
        return {"category": 2, "category_name": CATEGORIES[2], "confidence": "medium", "guard_type": gtype,
                "guard_value": gval, "flags": flags, "inferred_guard": inferred,
                "reason": "operator reported the drop before the automatic stop; the guard ended a run without the cube"}
      flags.append(f"drop_coincident_with_guard(drop {drop_t:g} s, stop {dur:.1f} s)")
      return {"category": 3, "category_name": CATEGORIES[3], "confidence": "low", "guard_type": gtype,
              "guard_value": gval, "flags": flags, "inferred_guard": inferred,
              "reason": "guard stop; operator drop time within the stop time +- margin, order unresolved"}
    return {"category": 3, "category_name": CATEGORIES[3], "confidence": "high" if gtype != "unrecorded" else "medium",
            "guard_type": gtype, "guard_value": gval, "flags": flags, "inferred_guard": inferred,
            "reason": "run ended by an automatic guard"}
  if run.stop_reason == "max_steps":
    if row is None:
      return {"category": 1, "category_name": CATEGORIES[1], "confidence": "low", "guard_type": None,
              "guard_value": float("nan"), "flags": ["no_observer_record"], "reason": "ran to the end; drop not observed"}
    if dropped:
      return {"category": 2, "category_name": CATEGORIES[2], "confidence": "high", "guard_type": None,
              "guard_value": float("nan"), "flags": ["ran_to_end_after_drop"], "reason": "operator reported a drop"}
    return {"category": 1, "category_name": CATEGORIES[1], "confidence": "high", "guard_type": None,
            "guard_value": float("nan"), "flags": [], "reason": "ran 60 s, no drop reported"}
  if run.stop_reason == "interrupted":
    if row is None:
      return {"category": 5, "category_name": CATEGORIES[5], "confidence": "high", "guard_type": None,
              "guard_value": float("nan"), "flags": ["not_in_trial_record"],
              "reason": "stopped by the operator (Ctrl+C); not entered in trials.csv; reason not recorded"}
    if dropped:
      return {"category": 2, "category_name": CATEGORIES[2], "confidence": "high", "guard_type": None,
              "guard_value": float("nan"), "flags": ["operator_stopped_after_drop"], "reason": "operator reported a drop"}
    if (row.get("notes") or "").strip():
      return {"category": 4, "category_name": CATEGORIES[4], "confidence": "medium", "guard_type": None,
              "guard_value": float("nan"), "flags": [], "reason": f"operator stop, note: {row['notes']}"}
    return {"category": 5, "category_name": CATEGORIES[5], "confidence": "high", "guard_type": None,
            "guard_value": float("nan"), "flags": ["recorded_without_reason"], "reason": "operator stop without a reason"}
  return {"category": 6, "category_name": CATEGORIES[6], "confidence": "low", "guard_type": None,
          "guard_value": float("nan"), "flags": [f"unknown_stop_reason:{run.stop_reason}"], "reason": "unrecognised stop reason"}


# ----------------------------------------------------------------------------------- signals
def duration_s(run: RunLog) -> float:
  t = run.f.get("elapsed_s")
  return float(t[-1] - t[0]) if t is not None and len(t) > 1 and np.all(np.isfinite(t[[0, -1]])) else float("nan")


def times(run: RunLog) -> tuple[np.ndarray | None, np.ndarray | None]:
  """(t_read, t_cmd) relative to the first policy-loop state read; None if unusable."""
  t, c = run.f.get("elapsed_s"), run.f.get("loop_compute_s")
  if t is None or "timestamps_nonfinite" in run.flags:
    return None, None
  if c is None or not np.all(np.isfinite(c)):
    c = np.zeros_like(t)  # t_read unknown -> use write time; flagged by the caller
  tr = t - c
  return tr - tr[0], t - tr[0]


def tracking_errors(q: np.ndarray, cmd: np.ndarray) -> np.ndarray:
  """Signed per-joint error command[k-1] - q[k]; row 0 is NaN (previous command not logged)."""
  e = np.full(q.shape, np.nan)
  e[1:] = cmd[:-1] - q[1:]
  return e


def rowmax(x: np.ndarray) -> np.ndarray:
  """max over axis 1 ignoring NaN; all-NaN rows -> NaN (no warning)."""
  m = np.max(np.where(np.isfinite(x), x, -np.inf), axis=1)
  return np.where(np.isfinite(m), m, np.nan)


def pct(a: np.ndarray, p: float) -> float:
  a = a[np.isfinite(a)]
  return float(np.percentile(a, p)) if a.size else float("nan")


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
  """[start, end) index pairs of consecutive True."""
  if mask.size == 0:
    return []
  m = np.concatenate([[False], mask.astype(bool), [False]])
  dm = np.diff(m.astype(np.int8))
  return list(zip(np.flatnonzero(dm == 1).tolist(), np.flatnonzero(dm == -1).tolist()))


def run_durations(mask: np.ndarray, t: np.ndarray, dt_nom: float) -> list[float]:
  return [float(t[b - 1] - t[a] + dt_nom) for a, b in runs_of(mask)]


def jam_episodes(blocked: np.ndarray, t: np.ndarray, dt_nom: float) -> list[dict[str, Any]]:
  """Jam episodes.  A sample is in a jam when, in the centred JAM_WINDOW_S window around it, one
  joint is blocked (|I| >= SAT_MA and |error| >= JAM_TRACKING_RAD) for >= JAM_MIN_FRAC of the
  samples.  Gaps shorter than half a window are merged; episodes shorter than SUSTAINED_S are
  dropped.  'unresolved' = the episode reaches the end of the log (within half a window)."""
  n = blocked.shape[0]
  k = max(1, int(round(JAM_WINDOW_S / dt_nom)))
  if n < k:
    return []
  ker = np.ones(k)
  cnt = np.convolve(np.ones(n), ker, "same")  # real samples per window (no zero padding at the ends)
  frac = np.stack([np.convolve(blocked[:, j].astype(float), ker, "same") / cnt for j in range(16)], axis=1)
  state = (frac >= JAM_MIN_FRAC).any(1)
  spans = runs_of(state)
  merged: list[list[int]] = []
  for a, b in spans:
    if merged and a - merged[-1][1] < k // 2:
      merged[-1][1] = b
    else:
      merged.append([a, b])
  eps = []
  for a, b in merged:
    dur = float(t[b - 1] - t[a] + dt_nom)
    if dur < SUSTAINED_S:
      continue
    js = sorted({HW_JOINTS[j] for j in range(16) if blocked[a:b, j].mean() >= 0.25})
    eps.append({"start_s": float(t[a]), "end_s": float(t[b - 1] + dt_nom), "duration_s": dur, "joints": js,
                "unresolved": bool(b >= n - k // 2)})
  return eps


def current_metrics(I: np.ndarray, t: np.ndarray, dt_nom: float) -> dict[str, Any]:
  """Per-joint |current| statistics (I: (n,16) signed mA)."""
  A = np.abs(I)
  out: dict[str, Any] = {"per_joint": {}}
  for j, nm in enumerate(HW_JOINTS):
    a = A[:, j]
    fin = np.isfinite(a)
    if not fin.any():
      out["per_joint"][nm] = None
      continue
    k = int(np.nanargmax(a))
    pj: dict[str, Any] = {"mean": float(np.nanmean(a)), "median": pct(a, 50), "p95": pct(a, 95), "p99": pct(a, 99),
                          "max": float(a[k]), "peak_step": k, "peak_t_s": float(t[k])}
    for lv in CURRENT_LEVELS_MA:
      m = a >= lv
      d = run_durations(m, t, dt_nom)
      pj[f"n_ge_{lv:.0f}"] = int(m.sum())
      pj[f"frac_ge_{lv:.0f}"] = float(m.mean())
      pj[f"longest_ge_{lv:.0f}_s"] = max(d) if d else 0.0
      pj[f"episodes_ge_{lv:.0f}_over_{SUSTAINED_S}s"] = int(sum(x >= SUSTAINED_S for x in d))
    out["per_joint"][nm] = pj
  return out


def finger_metrics(I: np.ndarray) -> dict[str, Any]:
  A = np.abs(I)
  tot = np.nansum(A, axis=1)
  out = {}
  for fn, js in FINGERS.items():
    s = np.nansum(A[:, js], axis=1)
    out[fn] = {"mean": float(np.mean(s)), "p95": pct(s, 95), "p99": pct(s, 99), "max": float(np.max(s)),
               "share_of_total": float(np.sum(s) / max(np.sum(tot), 1e-9)),
               "top_joint": HW_JOINTS[js[int(np.argmax(np.nanmean(A[:, js], axis=0)))]],
               "top_joint_share_of_finger": float(np.nanmax(np.nanmean(A[:, js], axis=0)) / max(np.nanmean(s), 1e-9))}
  return out


def stats(a: np.ndarray) -> dict[str, float]:
  a = a[np.isfinite(a)]
  if not a.size:
    return {"mean": float("nan"), "median": float("nan"), "p95": float("nan"), "p99": float("nan"), "max": float("nan")}
  return {"mean": float(a.mean()), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
          "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def lag_position(t_read: np.ndarray, q: np.ndarray, t_cmd: np.ndarray, c: np.ndarray, mask: np.ndarray,
                 grid: np.ndarray = LAG_GRID_S) -> tuple[float, float]:
  """tau minimising mean |q(t_read) - c_lin(t_read - tau)| (c linearly interpolated between writes).
  Returns (tau, cost at optimum / cost at tau=0)."""
  cost = []
  for tau in grid:
    w = mask & (t_read - tau >= t_cmd[0])
    if w.sum() < LAG_MIN_SAMPLES:
      cost.append(np.nan)
      continue
    cost.append(np.mean(np.abs(q[w] - np.interp(t_read[w] - tau, t_cmd, c))))
  cost = np.array(cost)
  if not np.isfinite(cost).any():
    return float("nan"), float("nan")
  i = int(np.nanargmin(cost))
  return float(grid[i]), float(cost[i] / cost[0]) if np.isfinite(cost[0]) and cost[0] > 0 else float("nan")


def lag_velocity(t_read: np.ndarray, v: np.ndarray, t_cmd: np.ndarray, c: np.ndarray, mask: np.ndarray,
                 grid: np.ndarray = LAG_GRID_S) -> tuple[float, float]:
  """tau maximising corr(v(t_read), command rate at t_read - tau); offset-insensitive."""
  dtc = np.diff(t_cmd)
  if np.any(dtc <= 0):
    return float("nan"), float("nan")
  vc = np.diff(c) / dtc
  tm = 0.5 * (t_cmd[1:] + t_cmd[:-1])
  r = []
  for tau in grid:
    w = mask & (t_read - tau >= tm[0]) & np.isfinite(v)
    if w.sum() < LAG_MIN_SAMPLES:
      r.append(np.nan)
      continue
    x = np.interp(t_read[w] - tau, tm, vc)
    r.append(np.corrcoef(x, v[w])[0, 1] if np.std(x) > 0 and np.std(v[w]) > 0 else np.nan)
  r = np.array(r)
  if not np.isfinite(r).any():
    return float("nan"), float("nan")
  i = int(np.nanargmax(r))
  return float(grid[i]), float(r[i])


def fd_velocity(t: np.ndarray, q: np.ndarray) -> np.ndarray:
  """Finite-difference velocity at sample times (central where possible), using real timestamps."""
  return np.gradient(q, t, axis=0)


# ------------------------------------------------------------------------------- per attempt
def analyse_run(run: RunLog, cfg: dict[str, float]) -> dict[str, Any]:
  res: dict[str, Any] = {}
  if not run.ok or run.n < 2:
    return res
  tr, tc = times(run)
  if tr is None:
    return res
  if run.f["loop_compute_s"] is None:
    run.flags.append("t_read_unavailable_used_write_time")
  q, qd, I, c = run.f["q_hw"], run.f["qd_hw"], run.f["current_ma"], run.f["command_hw"]
  temp = run.f["temperature_c"].astype(np.float64)
  dt = np.diff(run.f["elapsed_s"])
  valid_dt = dt[np.isfinite(dt) & (dt > 0)]
  dt_nom = float(np.median(valid_dt)) if valid_dt.size else 0.05
  # --- timing
  lc = run.f["loop_compute_s"]
  res["timing"] = {**{f"interval_{k}_ms": v * 1000 for k, v in stats(valid_dt).items()},
                   "frac_gt_50ms": float(np.mean(valid_dt > 0.050)) if valid_dt.size else float("nan"),
                   "frac_gt_60ms": float(np.mean(valid_dt > 0.060)) if valid_dt.size else float("nan"),
                   "frac_gt_100ms": float(np.mean(valid_dt > 0.100)) if valid_dt.size else float("nan"),
                   "n_intervals": int(valid_dt.size), "n_invalid_intervals": int(dt.size - valid_dt.size),
                   **({f"compute_{k}_ms": v * 1000 for k, v in stats(lc).items()} if lc is not None else {})}
  last2 = tc[1:] >= tc[-1] - 2.0
  if valid_dt.size == dt.size and last2.any() and (~last2).any():
    res["timing"]["frac_gt_60ms_last_2s"] = float(np.mean(dt[last2] > 0.060))
    res["timing"]["frac_gt_60ms_before_last_2s"] = float(np.mean(dt[~last2] > 0.060))
  # --- current
  res["current"] = current_metrics(I, tr, dt_nom)
  res["finger"] = finger_metrics(I)
  res["peak_current_any_joint"] = float(np.nanmax(np.abs(I)))
  res["peak_current_joint"] = HW_JOINTS[int(np.nanargmax(np.nanmax(np.abs(I), axis=0)))]
  res["peak_temperature_c"] = float(np.nanmax(temp)) if np.isfinite(temp).any() else float("nan")
  # --- tracking
  e = tracking_errors(q, c)
  ae = np.abs(e)
  logged = run.f["tracking_error_rad"]
  if logged is not None and run.n > 1:
    rec = rowmax(ae[1:])
    res["tracking_logged_reproduced_max_abs_diff"] = float(np.max(np.abs(rec - logged[1:])))
  res["tracking"] = {nm: stats(ae[:, j]) for j, nm in enumerate(HW_JOINTS)}
  res["tracking_max_any"] = float(np.nanmax(ae)) if np.isfinite(ae).any() else float("nan")
  res["tracking_logged_max"] = float(np.nanmax(logged)) if logged is not None else float("nan")
  # --- saturation / jam
  A = np.abs(I)
  sat = A >= SAT_MA
  jam = sat & (ae >= JAM_TRACKING_RAD)
  res["saturation"] = {"frac_any_joint": float(sat.any(1).mean()),
                       "per_joint_frac": {nm: float(sat[:, j].mean()) for j, nm in enumerate(HW_JOINTS)},
                       "jam_frac_per_joint": {nm: float(jam[:, j].mean()) for j, nm in enumerate(HW_JOINTS)},
                       "index_frac": float(sat[:, 1:3].any(1).mean()), "thumb_axl_frac": float(sat[:, 13].mean()),
                       "longest_sat_any_joint_s": max([0.0] + [max(run_durations(sat[:, j], tr, dt_nom) or [0.0]) for j in range(16)]),
                       "index_blocked_frac": float(jam[:, 1:3].any(1).mean())}
  eps = jam_episodes(jam, tr, dt_nom)
  res["jam_episodes"] = eps
  res["jam_summary"] = {"n_episodes": len(eps), "total_s": float(sum(e["duration_s"] for e in eps)),
                        "ended_in_jam": bool(eps and eps[-1]["unresolved"]),
                        "resolved_durations_s": [e["duration_s"] for e in eps if not e["unresolved"]],
                        "first_start_s": eps[0]["start_s"] if eps else None,
                        "index_involved": bool(any(set(e["joints"]) & {"if_mcp", "if_pip"} for e in eps))}
  # --- windows (policy time)
  win = {}
  for a, b in WINDOWS_S:
    w = (tr >= a) & (tr < b)
    if w.sum() < 10:
      continue
    win[f"{a}-{b}"] = {"n": int(w.sum()), "index_sat_frac": float(sat[w][:, 1:3].any(1).mean()),
                       "thumb_axl_sat_frac": float(sat[w][:, 13].mean()), "any_sat_frac": float(sat[w].any(1).mean()),
                       "jam_any_frac": float(jam[w].any(1).mean()),
                       **{f"{fn}_mean_ma": float(np.nanmean(np.nansum(A[w][:, js], axis=1))) for fn, js in FINGERS.items()},
                       "tracking_p95_any": pct(rowmax(ae[w]), 95),
                       "mean_abs_qd": float(np.nanmean(np.abs(qd[w])))}
  res["windows"] = win
  # --- movement
  dc = np.abs(np.diff(c, axis=0))
  dtc = np.diff(tc)[:, None]
  cr = np.where(dtc > 0, dc / np.where(dtc > 0, dtc, 1.0), np.nan)  # invalid intervals -> NaN, not inf
  near_lo = c <= HW_LIMIT_LO + LIMIT_MARGIN_RAD + 0.02
  near_hi = c >= HW_LIMIT_HI - LIMIT_MARGIN_RAD - 0.02
  fdv = fd_velocity(tr, q) if "timestamps_nonmonotonic" not in run.flags else np.full(q.shape, np.nan)
  res["movement"] = {"abs_qd": {nm: stats(np.abs(qd[:, j])) for j, nm in enumerate(HW_JOINTS)},
                     "abs_qd_mean_all": float(np.nanmean(np.abs(qd))),
                     "command_rate": {nm: stats(cr[:, j]) for j, nm in enumerate(HW_JOINTS)},
                     "command_rate_mean_all": float(np.nanmean(cr)),
                     "frac_command_steps_at_max_delta": float(np.mean(dc >= 0.99 * COMMAND_DELTA_MAX)),
                     "per_joint_frac_at_max_delta": {nm: float(np.mean(dc[:, j] >= 0.99 * COMMAND_DELTA_MAX)) for j, nm in enumerate(HW_JOINTS)},
                     "frac_command_near_limit": {nm: float(np.mean(near_lo[:, j] | near_hi[:, j])) for j, nm in enumerate(HW_JOINTS)},
                     "command_reversals_per_s": float(np.mean(np.sum(np.diff(np.sign(np.diff(c, axis=0)), axis=0) != 0, axis=0)) / max(tc[-1], 1e-9)),
                     "max_command_step_rad": float(np.max(dc)) if dc.size else float("nan"),
                     "q_excursion_rad": {nm: float(np.nanmax(q[:, j]) - np.nanmin(q[:, j])) for j, nm in enumerate(HW_JOINTS)}}
  # qd_hw vs finite difference of q_hw
  ok = np.isfinite(fdv) & np.isfinite(qd)
  res["qd_check"] = {"corr": float(np.corrcoef(fdv[ok], qd[ok])[0, 1]) if ok.sum() > 10 else float("nan"),
                     "median_abs_diff_rad_s": float(np.median(np.abs(fdv[ok] - qd[ok]))) if ok.any() else float("nan"),
                     "std_ratio_fd_over_qd": float(np.std(fdv[ok]) / max(np.std(qd[ok]), 1e-9)) if ok.any() else float("nan")}
  # --- lag (unsaturated samples only, runs >= 10 s)
  lag = {}
  if tr[-1] >= 10.0 and "timestamps_nonmonotonic" not in run.flags:
    for j, nm in enumerate(HW_JOINTS):
      m = np.isfinite(A[:, j]) & (A[:, j] < LAG_MAX_CURRENT_MA)
      tp, cr_ = lag_position(tr, q[:, j], tc, c[:, j], m)
      tv, rv = lag_velocity(tr, qd[:, j], tc, c[:, j], m)
      tf, rf = lag_velocity(tr, fdv[:, j], tc, c[:, j], m)
      lag[nm] = {"pos_s": tp, "pos_cost_ratio": cr_, "vel_qd_s": tv, "vel_qd_corr": rv, "vel_fd_s": tf, "vel_fd_corr": rf,
                 "n": int(m.sum())}
  res["lag"] = lag
  # --- relationships (pooled per attempt)
  ve = np.abs(qd[1:]).ravel()
  ee = ae[1:].ravel()
  ce = A[1:].ravel()
  okr = np.isfinite(ve) & np.isfinite(ee) & np.isfinite(ce)
  res["relationships"] = {"spearman_abs_qd_vs_tracking": spearman(ve[okr], ee[okr]),
                          "spearman_abs_qd_vs_current": spearman(ve[okr], ce[okr]),
                          "spearman_tracking_vs_current": spearman(ee[okr], ce[okr])}
  res["dt_nominal_s"] = dt_nom
  return res


def spearman(x: np.ndarray, y: np.ndarray) -> float:
  if x.size < 10:
    return float("nan")
  rx = np.argsort(np.argsort(x)).astype(float)
  ry = np.argsort(np.argsort(y)).astype(float)
  return float(np.corrcoef(rx, ry)[0, 1])


def current_vs_error(runs: list[RunLog]) -> dict[str, Any]:
  """Pooled quasi-static |current| vs |tracking error| per joint: slope (mA/rad, |e| < 0.1 rad)
  and the error at which the current reaches the limit (limit / slope)."""
  out = {}
  E, C, V = [], [], []
  for r in runs:
    if not r.ok or r.n < 3:
      continue
    e = tracking_errors(r.f["q_hw"], r.f["command_hw"])[1:]
    E.append(e)
    C.append(r.f["current_ma"][1:])
    V.append(r.f["qd_hw"][1:])
  if not E:
    return out
  E, C, V = np.concatenate(E), np.concatenate(C), np.concatenate(V)
  bins = np.array([0, .02, .05, .08, .11, .15, .2, .25, .3, .4, .6])
  for j, nm in enumerate(HW_JOINTS):
    m = (np.abs(V[:, j]) < QUASI_STATIC_RAD_S) & np.isfinite(E[:, j]) & np.isfinite(C[:, j])
    s = m & (np.abs(E[:, j]) < 0.1)
    slope = float(np.polyfit(E[s, j], C[s, j], 1)[0]) if s.sum() > 50 else float("nan")
    r = float(np.corrcoef(E[s, j], C[s, j])[0, 1]) if s.sum() > 50 else float("nan")
    med = []
    for a, b in zip(bins[:-1], bins[1:]):
      w = m & (np.abs(E[:, j]) >= a) & (np.abs(E[:, j]) < b)
      med.append(float(np.median(C[w, j] * np.sign(E[w, j]))) if w.sum() > 30 else float("nan"))
    out[nm] = {"slope_ma_per_rad": slope, "corr": r, "bins_rad": bins.tolist(), "median_current_along_error_ma": med,
               "n_quasi_static": int(m.sum())}
  return out


# ---------------------------------------------------------------------------- guard events
def guard_event(run: RunLog, att: dict, cfg: dict[str, float]) -> dict[str, Any]:
  tr, tc = times(run)
  q, qd, I, c = run.f["q_hw"], run.f["qd_hw"], run.f["current_ma"], run.f["command_hw"]
  A = np.abs(I)
  e = np.abs(tracking_errors(q, c))
  dt_nom = att["metrics"].get("dt_nominal_s", 0.05)
  last = run.n - 1
  w1 = tr >= tr[-1] - 1.0
  w2 = tr >= tr[-1] - 2.0
  mean1 = np.nanmean(A[w1], axis=0)
  fin_last = {fn: float(np.nansum(A[last, js])) for fn, js in FINGERS.items()}
  sat2 = (A[w2] >= SAT_MA).mean(0)
  dtw = np.diff(run.f["elapsed_s"])[w2[1:]]
  js = att["metrics"].get("jam_summary", {})
  if js.get("ended_in_jam"):
    trip_class = "sustained_jam"
  elif tr[-1] + dt_nom <= START_TRANSIENT_S and float(np.nanmax(A[last])) < SAT_MA:
    trip_class = "start_transient"
  else:
    trip_class = "other"
  return {"attempt": run.name, "trip_class": trip_class,
          "trip_jump_from_last_logged_ma": float(att["guard_value"] - np.nanmax(A[last])) if att["guard_type"] == "current" else None, "policy": att["policy"], "category": att["category_name"], "stop_reason": run.stop_reason,
          "guard_type": att["guard_type"], "trip_value": att["guard_value"], "inferred_guard": att.get("inferred_guard"),
          "last_logged_t_s": float(tr[-1]), "estimated_trip_t_s": float(tr[-1] + dt_nom),
          "note": "the tripping read is not logged; values below end one control period before the trip",
          "joint_highest_current_last_sample": HW_JOINTS[int(np.nanargmax(A[last]))],
          "joint_highest_mean_current_last_1s": HW_JOINTS[int(np.nanargmax(mean1))],
          "finger_highest_current_last_sample": max(fin_last, key=fin_last.get),
          "current_last_sample_ma": {nm: float(A[last, j]) for j, nm in enumerate(HW_JOINTS)},
          "finger_current_last_sample_ma": fin_last,
          "peak_logged_current_ma": float(np.nanmax(A)), "peak_logged_joint": HW_JOINTS[int(np.nanargmax(np.nanmax(A, axis=0)))],
          "sat_frac_last_2s": {nm: float(sat2[j]) for j, nm in enumerate(HW_JOINTS) if sat2[j] > 0},
          "tracking_last_sample_rad": {nm: float(e[last, j]) for j, nm in enumerate(HW_JOINTS)},
          "tracking_max_last_2s_rad": float(np.nanmax(e[w2])), "tracking_joint_max_last_2s": HW_JOINTS[int(np.nanargmax(np.nanmax(e[w2], axis=0)))],
          "tracking_history_last_2s_rad": [float(x) for x in rowmax(e[w2])],
          "command_minus_measured_last_sample_rad": {nm: float(c[last - 1, j] - q[last, j]) for j, nm in enumerate(HW_JOINTS)} if last > 0 else {},
          "abs_qd_last_sample": {nm: float(abs(qd[last, j])) for j, nm in enumerate(HW_JOINTS)},
          "max_abs_qd_last_2s": float(np.nanmax(np.abs(qd[w2]))),
          "interval_ms_last_2s": {"median": float(np.median(dtw) * 1000) if dtw.size else float("nan"),
                                  "max": float(np.max(dtw) * 1000) if dtw.size else float("nan"),
                                  "frac_gt_60ms": float(np.mean(dtw > 0.06)) if dtw.size else float("nan")},
          "index_sat_by_window": {k: v["index_sat_frac"] for k, v in att["metrics"].get("windows", {}).items()}}


# ---------------------------------------------------------------------------- quarter turns
def quarter_turns(row: dict | None, run_s: float) -> dict[str, Any]:
  """Operator counts are quarter-turns (90 deg).  full_revolutions = quarter_turns / 4."""
  if row is None:
    return {"observed_quarter_turns": None, "full_revolutions": None, "held_s": None, "full_rev_per_min_held": None}
  qt = _f(row.get("observed_turns"))
  dropped = row.get("dropped") == "True"
  dt = _f(row.get("drop_time_s"))
  held = min(dt, run_s) if dropped and math.isfinite(dt) else (float("nan") if dropped else run_s)
  rev = qt / 4.0 if math.isfinite(qt) else float("nan")
  rpm = rev / held * 60.0 if math.isfinite(rev) and math.isfinite(held) and held >= 20.0 else float("nan")
  return {"observed_quarter_turns": qt, "full_revolutions": rev, "held_s": held, "full_rev_per_min_held": rpm}


# ------------------------------------------------------------------------------ simulation
def sim_metrics(path: Path, hw_sat_err: dict[str, float], n_envs_lag: int = 32) -> dict[str, Any]:
  """Same tracking / lag / movement metrics on a simulation rollout (sim_actuator_reference.py)."""
  d = np.load(path, allow_pickle=False)
  names = [str(x) for x in d["joint_names"]]
  if tuple(names) != SERVER_JOINTS:
    raise ValueError(f"unexpected sim joint order {names}")
  q = d["q"][..., SERVER_TO_HW].astype(np.float64)   # -> hardware order
  c = d["cmd"][..., SERVER_TO_HW].astype(np.float64)
  F = d["force"][..., SERVER_TO_HW].astype(np.float64)
  L = d["effort_limit"][..., SERVER_TO_HW].astype(np.float64)
  alive = d["alive"]
  dt = float(d["dt"])
  T, n, _ = q.shape
  t = np.arange(T) * dt
  e = np.abs(c[:-1] - q[1:])
  al = alive[1:]
  out: dict[str, Any] = {"policy": str(d["policy"]), "n_envs": n, "steps": T, "dt": dt, "alive_frac_end": float(alive[-1].mean())}
  out["tracking"] = {nm: stats(e[..., j][al]) for j, nm in enumerate(HW_JOINTS)}
  out["frac_tracking_gt_hw_saturation_error"] = {nm: float(np.mean(e[..., j][al] > hw_sat_err.get(nm, np.nan)))
                                                 for j, nm in enumerate(HW_JOINTS) if math.isfinite(hw_sat_err.get(nm, np.nan))}
  r = np.abs(F) / L[None]
  out["force_over_limit"] = {nm: {"p99": pct(r[..., j][alive], 99), "max": float(np.nanmax(r[..., j])),
                                  "frac_ge_0.98": float(np.mean(r[..., j][alive] >= 0.98))} for j, nm in enumerate(HW_JOINTS)}
  out["effort_limit_nm"] = {"min": float(L.min()), "max": float(L.max())}
  # saturation error from the sim's own force-vs-error slope (small errors) and its effort limit
  sat_err = {}
  for j, nm in enumerate(HW_JOINTS):
    ee = (c[:-1, :, j] - q[1:, :, j])[al]
    ff = F[1:, :, j][al]
    s = np.abs(ee) < 0.1
    slope = np.polyfit(ee[s], ff[s], 1)[0] if s.sum() > 100 else np.nan
    sat_err[nm] = {"force_slope_nm_per_rad": float(slope), "saturation_error_rad_median_env":
                   float(np.median(L[:, j]) / slope) if slope and math.isfinite(slope) and slope > 0 else float("nan")}
  out["saturation_error"] = sat_err
  dc = np.abs(np.diff(c, axis=0))
  out["frac_command_steps_at_max_delta"] = float(np.mean(dc[al] >= 0.99 * COMMAND_DELTA_MAX))
  v = np.abs(np.gradient(q, t, axis=0))
  out["abs_qfd_mean_all"] = float(np.mean(v[alive]))
  out["abs_qfd"] = {nm: stats(v[..., j][alive]) for j, nm in enumerate(HW_JOINTS)}
  out["command_rate_mean_all"] = float(np.mean((dc / dt)[al]))
  lag: dict[str, Any] = {}
  envs = [i for i in range(n) if alive[:, i].all()][:n_envs_lag]
  for j, nm in enumerate(HW_JOINTS):
    tp, tv = [], []
    for i in envs:
      m = np.ones(T, bool)
      tp.append(lag_position(t, q[:, i, j], t, c[:, i, j], m)[0])
      tv.append(lag_velocity(t, np.gradient(q[:, i, j], t), t, c[:, i, j], m)[0])
    lag[nm] = {"pos_s_median": float(np.nanmedian(tp)), "pos_s_iqr": [pct(np.array(tp), 25), pct(np.array(tp), 75)],
               "vel_fd_s_median": float(np.nanmedian(tv)), "vel_fd_s_iqr": [pct(np.array(tv), 25), pct(np.array(tv), 75)],
               "n_envs": len(envs)}
  out["lag"] = lag
  return out


# ---------------------------------------------------------------------------------- driver
def audit(date: str, logs_dir: Path, csv_path: Path, out_dir: Path, sim_dir: Path | None = None,
          figures: bool = True) -> dict[str, Any]:
  env = read_hand_env()
  cfg = {"current_limit_ma": _f(env.get("CURRENT_LIMIT_MA")), "current_abort_ma": _f(env.get("CURRENT_ABORT_MA")),
         "tracking_error_abort_rad": _f(env.get("TRACKING_ERROR_ABORT_RAD")),
         "temperature_abort_c": _f(env.get("TEMPERATURE_ABORT_C")), "velocity_abort_rad_s": _f(env.get("VELOCITY_ABORT_RAD_S")),
         "max_command_step_rad": _f(env.get("MAX_COMMAND_STEP_RAD")), "gain_profile": env.get("LEAP_GAIN_PROFILE")}
  files = sorted(p for p in logs_dir.glob(f"physical_{date}_*.npz") if "_part" not in p.stem)
  parts = sorted(p for p in logs_dir.glob(f"physical_{date}_*_part*.npz"))
  runs = [load_run(p) for p in files]
  rows = read_trials(csv_path)
  rowmap, orphans, dups = match_trials(runs, rows)
  sha_by_policy: dict[str, set[str]] = {}
  for r in rows:
    sha_by_policy.setdefault(r["policy"], set()).add(r.get("checkpoint_sha256", ""))
  attempts = []
  for run in runs:
    row = rowmap.get(run.name)
    pol, pnotes = identify_policy(run, row, sha_by_policy)
    cl = classify(run, row, cfg["tracking_error_abort_rad"] if math.isfinite(cfg["tracking_error_abort_rad"]) else 0.65)
    met = analyse_run(run, cfg)
    dur = duration_s(run)
    att = {"file": run.name, "policy": pol, "checkpoint_sha256": run.meta.get("policy_sha256"),
           "policy_iteration": run.meta.get("policy_iteration"), "started_at": run.started_at,
           "trial": int(row["trial"]) if row else None, "csv_time": row["time"] if row else None,
           "duration_s": dur, "n_steps": run.n, "completed": run.completed, "stop_reason": run.stop_reason,
           **cl, "policy_notes": pnotes, "data_quality_flags": list(run.flags),
           "operator": {k: row.get(k) for k in ("observed_turns", "dropped", "drop_time_s", "intervened", "notes",
                                                "max_current_ma", "jitter_frac_gt_60ms")} if row else None,
           "turns": quarter_turns(row, dur), "metrics": met}
    att["peak_abs_current_ma"] = met.get("peak_current_any_joint", float("nan"))
    att["peak_temperature_c"] = met.get("peak_temperature_c", float("nan"))
    att["peak_tracking_error_rad"] = met.get("tracking_logged_max", float("nan"))
    if row and math.isfinite(_f(row.get("max_current_ma"))) and math.isfinite(att["peak_abs_current_ma"]):
      if abs(_f(row["max_current_ma"]) - att["peak_abs_current_ma"]) > 1.0:
        att["data_quality_flags"].append("csv_peak_current_differs")
    if row and abs(_f(row.get("run_s")) - dur) > 0.02:
      att["data_quality_flags"].append("csv_run_s_differs")
    attempts.append(att)
  cve = current_vs_error(runs)
  hw_sat_err = {nm: cfg["current_limit_ma"] / v["slope_ma_per_rad"] for nm, v in cve.items()
                if v["slope_ma_per_rad"] and math.isfinite(v["slope_ma_per_rad"]) and v["slope_ma_per_rad"] > 0}
  events = [guard_event(run, att, cfg) for run, att in zip(runs, attempts) if att["guard_type"] is not None and att["metrics"]]
  summary = summarise(attempts)
  recon = reconcile(attempts, rows, orphans, dups, files, parts)
  c5 = c5_comparison(attempts)
  sim = {}
  if sim_dir is not None and sim_dir.exists():
    for p in sorted(sim_dir.glob("sim_*.npz")):
      try:
        sim[p.stem] = sim_metrics(p, hw_sat_err)
      except Exception as exc:  # a broken sim file must not stop the hardware audit
        sim[p.stem] = {"error": str(exc)}
  lag_summary = summarise_lag(attempts)
  result = {"date": date, "config_hand_env": cfg, "n_files": len(files), "n_part_files": len(parts), "attempts": attempts,
            "reconciliation": recon, "policy_summary": summary, "guard_events": events, "c5_comparison": c5,
            "current_vs_error": cve, "hw_saturation_error_rad": hw_sat_err, "lag_summary": lag_summary,
            "sim_reference": sim}
  write_outputs(result, out_dir)
  if figures:
    make_figures(result, runs, out_dir / "figures")
  return result


def summarise(attempts: list[dict]) -> dict[str, Any]:
  pols = sorted({a["policy"] for a in attempts}, key=lambda p: ("A3", "C5", "R1", "P0", "M400").index(p) if p in ("A3", "C5", "R1", "P0", "M400") else 99)
  out = {}
  for p in pols:
    A = [a for a in attempts if a["policy"] == p]
    cats = {CATEGORIES[k]: sum(a["category"] == k for a in A) for k in CATEGORIES}
    rec = [a for a in A if a["trial"] is not None and a["operator"] and a["operator"].get("intervened") != "True"]
    qt = np.array([a["turns"]["observed_quarter_turns"] for a in rec if a["turns"]["observed_quarter_turns"] is not None
                   and math.isfinite(a["turns"]["observed_quarter_turns"])])
    rpm = np.array([a["turns"]["full_rev_per_min_held"] for a in rec if a["turns"]["full_rev_per_min_held"] is not None
                    and math.isfinite(a["turns"]["full_rev_per_min_held"])])
    held = np.array([a["turns"]["held_s"] if a["turns"]["held_s"] is not None and math.isfinite(a["turns"]["held_s"]) else 0.0 for a in rec])
    sat = [a["metrics"]["saturation"]["index_frac"] for a in A if a["metrics"]]
    out[p] = {"attempts": len(A), "categories": cats,
              "guard_events_any_category": int(sum(a["guard_type"] is not None for a in A)),
              "recorded_trials": len(rec),
              "quarter_turns_per_trial_mean": float(qt.mean()) if qt.size else float("nan"),
              "quarter_turns_per_trial_sd": float(qt.std(ddof=1)) if qt.size > 1 else float("nan"),
              "full_revolutions_per_trial_mean": float(qt.mean() / 4) if qt.size else float("nan"),
              "full_revolutions_per_trial_sd": float(qt.std(ddof=1) / 4) if qt.size > 1 else float("nan"),
              "n_counted": int(qt.size),
              "full_rev_per_min_held_mean": float(rpm.mean()) if rpm.size else float("nan"), "n_rate": int(rpm.size),
              "held_s_mean_recorded": float(held.mean()) if held.size else float("nan"),
              "index_sat_frac_median": float(np.median(sat)) if sat else float("nan"),
              "peak_current_max_ma": float(np.nanmax([a["peak_abs_current_ma"] for a in A])),
              "timing_frac_gt_60ms_mean": float(np.nanmean([a["metrics"]["timing"]["frac_gt_60ms"] for a in A if a["metrics"]])),
              "abs_qd_mean": float(np.nanmean([a["metrics"]["movement"]["abs_qd_mean_all"] for a in A if a["metrics"]])),
              "frac_command_at_max_delta_mean": float(np.nanmean([a["metrics"]["movement"]["frac_command_steps_at_max_delta"] for a in A if a["metrics"]]))}
  return out


def reconcile(attempts, rows, orphans, dups, files, parts) -> dict[str, Any]:
  cat_tot = {CATEGORIES[k]: sum(a["category"] == k for a in attempts) for k in CATEGORIES}
  return {"n_files": len(files), "n_attempts": len(attempts), "n_csv_rows": len(rows),
          "n_files_with_csv_row": sum(a["trial"] is not None for a in attempts),
          "n_files_without_csv_row": sum(a["trial"] is None for a in attempts),
          "csv_rows_without_file": orphans, "duplicate_csv_rows": dups, "part_files_ignored": [p.name for p in parts],
          "category_totals": cat_tot, "categories_sum": sum(cat_tot.values()),
          "sums_to_files": sum(cat_tot.values()) == len(files),
          "guard_events_total": sum(a["guard_type"] is not None for a in attempts)}


def c5_comparison(attempts: list[dict]) -> dict[str, Any]:
  C = [a for a in attempts if a["policy"] == "C5" and a["metrics"]]
  groups = {"completed": [a for a in C if a["category"] == 1], "guard_stop": [a for a in C if a["category"] == 3],
            "dropped": [a for a in C if a["category"] == 2]}
  out: dict[str, Any] = {"n": {k: len(v) for k, v in groups.items()}, "files": {k: [a["file"] for a in v] for k, v in groups.items()},
                         "by_window": {}}
  keys = ("index_sat_frac", "thumb_axl_sat_frac", "jam_any_frac", "index_mean_ma", "thumb_mean_ma", "middle_mean_ma",
          "ring_mean_ma", "tracking_p95_any", "mean_abs_qd")
  for a, b in WINDOWS_S:
    wk = f"{a}-{b}"
    out["by_window"][wk] = {g: {k: [x["metrics"]["windows"][wk][k] for x in v if wk in x["metrics"]["windows"]] for k in keys}
                            for g, v in groups.items()}
  out["whole_attempt"] = {g: {"index_sat_frac": [x["metrics"]["saturation"]["index_frac"] for x in v],
                              "frac_command_at_max_delta": [x["metrics"]["movement"]["frac_command_steps_at_max_delta"] for x in v],
                              "abs_qd_mean": [x["metrics"]["movement"]["abs_qd_mean_all"] for x in v],
                              "timing_frac_gt_60ms": [x["metrics"]["timing"]["frac_gt_60ms"] for x in v],
                              "duration_s": [x["duration_s"] for x in v]} for g, v in groups.items()}
  return out


def summarise_lag(attempts: list[dict]) -> dict[str, Any]:
  out = {}
  for p in sorted({a["policy"] for a in attempts}):
    L = [a["metrics"]["lag"] for a in attempts if a["policy"] == p and a["metrics"] and a["metrics"].get("lag")]
    if not L:
      continue
    pj = {}
    for nm in HW_JOINTS:
      for key in ("pos_s", "vel_qd_s", "vel_fd_s"):
        v = np.array([l[nm][key] for l in L if nm in l and math.isfinite(l[nm][key])])
        pj.setdefault(nm, {})[key] = {"median": float(np.median(v)) if v.size else float("nan"),
                                      "iqr": [pct(v, 25), pct(v, 75)], "n_runs": int(v.size)}
    allpos = np.array([l[nm]["pos_s"] for l in L for nm in HW_JOINTS if math.isfinite(l[nm]["pos_s"])])
    allvel = np.array([l[nm]["vel_qd_s"] for l in L for nm in HW_JOINTS if math.isfinite(l[nm]["vel_qd_s"])])
    out[p] = {"per_joint": pj, "pos_s_median_all": float(np.median(allpos)) if allpos.size else float("nan"),
              "pos_s_iqr_all": [pct(allpos, 25), pct(allpos, 75)],
              "vel_qd_s_median_all": float(np.median(allvel)) if allvel.size else float("nan"),
              "vel_qd_s_iqr_all": [pct(allvel, 25), pct(allvel, 75)], "n_runs": len(L)}
  return out


# ---------------------------------------------------------------------------------- output
def _clean(x: Any) -> Any:
  if isinstance(x, dict):
    return {str(k): _clean(v) for k, v in x.items()}
  if isinstance(x, (list, tuple)):
    return [_clean(v) for v in x]
  if isinstance(x, (np.floating, float)):
    return None if not math.isfinite(float(x)) else round(float(x), 6)
  if isinstance(x, (np.integer,)):
    return int(x)
  if isinstance(x, np.bool_):
    return bool(x)
  return x


def write_outputs(res: dict[str, Any], out: Path) -> None:
  out.mkdir(parents=True, exist_ok=True)
  (out / "audit.json").write_text(json.dumps(_clean(res), indent=1) + "\n")
  for k in ("policy_summary", "guard_events", "c5_comparison", "reconciliation", "current_vs_error", "lag_summary", "sim_reference"):
    (out / f"{k}.json").write_text(json.dumps(_clean(res[k]), indent=1) + "\n")
  dq = {a["file"]: {"flags": a["data_quality_flags"], "category_flags": a["flags"], "policy_notes": a["policy_notes"],
                    "tracking_reproduced_max_abs_diff": a["metrics"].get("tracking_logged_reproduced_max_abs_diff"),
                    "qd_check": a["metrics"].get("qd_check")} for a in res["attempts"]}
  (out / "data_quality.json").write_text(json.dumps(_clean(dq), indent=1) + "\n")
  cols = ("file", "policy", "trial", "category", "category_name", "confidence", "guard_type", "guard_value", "stop_reason",
          "completed", "duration_s", "n_steps", "peak_abs_current_ma", "peak_temperature_c", "peak_tracking_error_rad",
          "observed_quarter_turns", "full_revolutions", "held_s", "full_rev_per_min_held", "index_sat_frac",
          "frac_gt_60ms", "abs_qd_mean", "frac_command_at_max_delta", "checkpoint_sha256", "started_at", "flags")
  with open(out / "attempts.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(cols)
    for a in res["attempts"]:
      m = a["metrics"]
      w.writerow([a["file"], a["policy"], a["trial"], a["category"], a["category_name"], a["confidence"], a["guard_type"],
                  a["guard_value"], a["stop_reason"], a["completed"], f"{a['duration_s']:.3f}", a["n_steps"],
                  a["peak_abs_current_ma"], a["peak_temperature_c"], a["peak_tracking_error_rad"],
                  a["turns"]["observed_quarter_turns"], a["turns"]["full_revolutions"], a["turns"]["held_s"],
                  a["turns"]["full_rev_per_min_held"], m["saturation"]["index_frac"] if m else None,
                  m["timing"]["frac_gt_60ms"] if m else None, m["movement"]["abs_qd_mean_all"] if m else None,
                  m["movement"]["frac_command_steps_at_max_delta"] if m else None, a["checkpoint_sha256"], a["started_at"],
                  ";".join(a["flags"] + a["data_quality_flags"])])
  with open(out / "per_joint.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(("file", "policy", "category_name", "joint", "finger", "I_mean", "I_median", "I_p95", "I_p99", "I_max", "I_peak_t_s",
                "frac_ge_200", "frac_ge_300", "frac_ge_340", "longest_ge_340_s", "track_mean", "track_median", "track_p95",
                "track_p99", "track_max", "qd_mean", "qd_p95", "qd_max", "cmd_rate_mean", "frac_at_max_delta",
                "frac_cmd_near_limit", "lag_pos_s", "lag_vel_s"))
    for a in res["attempts"]:
      m = a["metrics"]
      if not m:
        continue
      for j, nm in enumerate(HW_JOINTS):
        cj, tj = m["current"]["per_joint"][nm], m["tracking"][nm]
        fn = next(f for f, js in FINGERS.items() if j in js)
        lg = m["lag"].get(nm, {})
        w.writerow((a["file"], a["policy"], a["category_name"], nm, fn, cj["mean"], cj["median"], cj["p95"], cj["p99"], cj["max"],
                    cj["peak_t_s"], cj["frac_ge_200"], cj["frac_ge_300"], cj["frac_ge_340"], cj["longest_ge_340_s"],
                    tj["mean"], tj["median"], tj["p95"], tj["p99"], tj["max"], m["movement"]["abs_qd"][nm]["mean"],
                    m["movement"]["abs_qd"][nm]["p95"], m["movement"]["abs_qd"][nm]["max"], m["movement"]["command_rate"][nm]["mean"],
                    m["movement"]["per_joint_frac_at_max_delta"][nm], m["movement"]["frac_command_near_limit"][nm],
                    lg.get("pos_s"), lg.get("vel_qd_s")))
  with open(out / "per_finger.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(("file", "policy", "category_name", "finger", "sum_I_mean", "sum_I_p95", "sum_I_p99", "sum_I_max", "share_of_total",
                "top_joint", "top_joint_share_of_finger"))
    for a in res["attempts"]:
      m = a["metrics"]
      if not m:
        continue
      for fn, v in m["finger"].items():
        w.writerow((a["file"], a["policy"], a["category_name"], fn, v["mean"], v["p95"], v["p99"], v["max"], v["share_of_total"],
                    v["top_joint"], v["top_joint_share_of_finger"]))
  with open(out / "policy_summary.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    keys = ["attempts"] + [f"cat_{c}" for c in CATEGORIES.values()] + [
      "guard_events_any_category", "recorded_trials", "quarter_turns_per_trial_mean", "quarter_turns_per_trial_sd",
      "full_revolutions_per_trial_mean", "full_revolutions_per_trial_sd", "n_counted", "full_rev_per_min_held_mean", "n_rate",
      "held_s_mean_recorded", "index_sat_frac_median", "peak_current_max_ma", "timing_frac_gt_60ms_mean", "abs_qd_mean",
      "frac_command_at_max_delta_mean"]
    w.writerow(["policy"] + keys)
    for p, v in res["policy_summary"].items():
      w.writerow([p] + [v["attempts"]] + [v["categories"][c] for c in CATEGORIES.values()] + [v[k] for k in keys[7:]])
  with open(out / "guard_events.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(("attempt", "policy", "category", "guard_type", "trip_value", "inferred_guard", "estimated_trip_t_s",
                "joint_highest_current_last_sample", "joint_highest_mean_current_last_1s", "finger_highest_current_last_sample",
                "peak_logged_current_ma", "tracking_max_last_2s_rad", "tracking_joint_max_last_2s", "max_abs_qd_last_2s",
                "interval_ms_last_2s_max", "sat_frac_last_2s"))
    for e in res["guard_events"]:
      w.writerow((e["attempt"], e["policy"], e["category"], e["guard_type"], e["trip_value"], e["inferred_guard"],
                  e["estimated_trip_t_s"], e["joint_highest_current_last_sample"], e["joint_highest_mean_current_last_1s"],
                  e["finger_highest_current_last_sample"], e["peak_logged_current_ma"], e["tracking_max_last_2s_rad"],
                  e["tracking_joint_max_last_2s"], e["max_abs_qd_last_2s"], e["interval_ms_last_2s"]["max"],
                  json.dumps(_clean(e["sat_frac_last_2s"]))))
  (out / "summary.md").write_text(summary_markdown(res))


def summary_markdown(res: dict[str, Any]) -> str:
  L = [f"# Hardware log audit {res['date']} (generated by kit/audit_hardware_logs.py)", "",
       f"Files: {res['n_files']}; CSV rows: {res['reconciliation']['n_csv_rows']}; categories sum to files: "
       f"{res['reconciliation']['sums_to_files']}.", "",
       "| policy | attempts | " + " | ".join(CATEGORIES.values()) + " | guard events (any) | quarter-turns/trial | full rev/trial | full rev/min held |",
       "|---|---|" + "---|" * len(CATEGORIES) + "---|---|---|---|"]
  for p, v in res["policy_summary"].items():
    L.append(f"| {p} | {v['attempts']} | " + " | ".join(str(v["categories"][c]) for c in CATEGORIES.values()) +
             f" | {v['guard_events_any_category']} | {v['quarter_turns_per_trial_mean']:.2f} (n={v['n_counted']}) | "
             f"{v['full_revolutions_per_trial_mean']:.2f} | {v['full_rev_per_min_held_mean']:.2f} (n={v['n_rate']}) |")
  L += ["", "| file | policy | trial | category | guard | duration s | peak I mA | index sat frac |", "|---|---|---|---|---|---|---|---|"]
  for a in res["attempts"]:
    m = a["metrics"]
    L.append(f"| {a['file']} | {a['policy']} | {a['trial'] or '-'} | {a['category_name']} | {a['guard_type'] or ''} {'' if a['guard_value'] is None or not math.isfinite(a['guard_value']) else a['guard_value']} | "
             f"{a['duration_s']:.1f} | {a['peak_abs_current_ma']:.0f} | {m['saturation']['index_frac']:.2f} |" if m else f"| {a['file']} | {a['policy']} | - | {a['category_name']} | | | | |")
  return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------- figures
POL_ORDER = ("A3", "C5", "R1", "P0", "M400")
POL_COL = {"A3": "#7a7a7a", "C5": "#2a9d8f", "R1": "#3b6fb6", "P0": "#e4572e", "M400": "#9b6fb3"}


def make_figures(res: dict[str, Any], runs: list[RunLog], fig_dir: Path) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig_dir.mkdir(parents=True, exist_ok=True)
  att = {a["file"]: a for a in res["attempts"]}
  pols = [p for p in POL_ORDER if p in res["policy_summary"]]

  def pooled(field_fn, pol):
    xs = [field_fn(r) for r in runs if att[r.name]["policy"] == pol and r.ok and r.n > 1]
    return np.concatenate(xs) if xs else np.zeros((0, 16))

  # 1/2: per-policy per-joint distributions of |current| and |tracking error|
  for name, fn, ylab, ref in (("fig_current_per_joint_by_policy.png", lambda r: np.abs(r.f["current_ma"]), "|current| [mA]",
                               (res["config_hand_env"]["current_limit_ma"], res["config_hand_env"]["current_abort_ma"])),
                              ("fig_tracking_per_joint_by_policy.png", lambda r: np.abs(tracking_errors(r.f["q_hw"], r.f["command_hw"]))[1:],
                               "|command(k-1) - measured(k)| [rad]", (res["config_hand_env"]["tracking_error_abort_rad"],))):
    fig, ax = plt.subplots(len(pols), 1, figsize=(15, 2.4 * len(pols)), sharex=True, sharey=True)
    for a, p in zip(np.atleast_1d(ax), pols):
      X = pooled(fn, p)
      a.boxplot([X[np.isfinite(X[:, j]), j] for j in range(16)], whis=(1, 99), showfliers=False, widths=0.6,
                patch_artist=True, boxprops={"facecolor": POL_COL[p], "alpha": 0.5})
      for rv in ref:
        a.axhline(rv, color="k", ls=":", lw=0.8)
      a.set_ylabel(f"{p}\n{ylab}", fontsize=8)
      for b in (4.5, 8.5, 12.5):
        a.axvline(b, color="0.8", lw=0.8)
    np.atleast_1d(ax)[-1].set_xticks(range(1, 17), HW_JOINTS, rotation=45, fontsize=8)
    fig.suptitle(f"{ylab} per joint, all attempts pooled per policy (box: 25-75%, whiskers 1-99%; dotted: kit limits)")
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=110)
    plt.close(fig)

  # 3/4: C5 completed vs guard-stop, time-matched windows
  c5 = res["c5_comparison"]
  wins = [f"{a}-{b}" for a, b in WINDOWS_S[:4]]
  fig, ax = plt.subplots(1, 4, figsize=(18, 4))
  for k, (key, ttl) in enumerate((("index_sat_frac", "index MCP/PIP at >= 340 mA (fraction)"),
                                  ("thumb_axl_sat_frac", "thumb axl at >= 340 mA (fraction)"),
                                  ("index_mean_ma", "index finger summed |current| [mA]"),
                                  ("thumb_mean_ma", "thumb summed |current| [mA]"))):
    for g, col in (("completed", "#2a9d8f"), ("guard_stop", "#c0392b"), ("dropped", "#f2a541")):
      for i, w in enumerate(wins):
        v = c5["by_window"][w][g][key]
        ax[k].scatter(np.full(len(v), i) + {"completed": -0.15, "guard_stop": 0.15, "dropped": 0.0}[g], v, color=col, s=18,
                      label=f"C5 {g} (n={c5['n'][g]})" if i == 0 else None)
    ax[k].set_xticks(range(len(wins)), [f"{w} s" for w in wins])
    ax[k].set_title(ttl, fontsize=9)
    ax[k].set_xlabel("time since policy start")
  ax[0].legend(fontsize=7)
  fig.suptitle("C5: completed vs guard-stop attempts, same time windows (one dot per attempt)")
  fig.tight_layout()
  fig.savefig(fig_dir / "fig_c5_current_completed_vs_guard.png", dpi=110)
  plt.close(fig)

  fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
  for g, col in (("completed", "#2a9d8f"), ("guard_stop", "#c0392b"), ("dropped", "#f2a541")):
    for f in c5["files"][g]:
      r = next(x for x in runs if x.name == f)
      tr, _ = times(r)
      e = np.abs(tracking_errors(r.f["q_hw"], r.f["command_hw"]))
      m = tr <= 16
      ax[0].plot(tr[m], rowmax(e[m][:, 1:3]), color=col, lw=0.7, alpha=0.8)
      ax[1].plot(tr[m], np.abs(r.f["current_ma"][m][:, 1:3]).max(1), color=col, lw=0.7, alpha=0.8)
  ax[0].axhline(res["config_hand_env"]["tracking_error_abort_rad"], color="k", ls=":")
  ax[1].axhline(res["config_hand_env"]["current_limit_ma"], color="k", ls=":")
  ax[0].set(title="C5 index MCP/PIP tracking error (max of the two)", xlabel="time since policy start [s]", ylabel="rad")
  ax[1].set(title="C5 index MCP/PIP |current| (max of the two)", xlabel="time since policy start [s]", ylabel="mA")
  from matplotlib.lines import Line2D
  ax[0].legend([Line2D([], [], color=c) for c in ("#2a9d8f", "#c0392b", "#f2a541")],
               [f"completed (n={c5['n']['completed']})", f"guard stop (n={c5['n']['guard_stop']})", f"dropped (n={c5['n']['dropped']})"], fontsize=8)
  fig.tight_layout()
  fig.savefig(fig_dir / "fig_c5_tracking_completed_vs_guard.png", dpi=110)
  plt.close(fig)

  # 5: joint velocity distributions + command rate
  fig, ax = plt.subplots(2, 1, figsize=(15, 6.5), sharex=True)
  w = 0.8 / len(pols)
  for i, p in enumerate(pols):
    V = pooled(lambda r: np.abs(r.f["qd_hw"]), p)
    Cr = pooled(lambda r: np.abs(np.diff(r.f["command_hw"], axis=0)) / np.diff(r.f["elapsed_s"])[:, None], p)
    for a, X in ((ax[0], V), (ax[1], Cr)):
      a.boxplot([X[:, j] for j in range(16)], positions=np.arange(16) + (i - (len(pols) - 1) / 2) * w, widths=w * 0.9,
                whis=(1, 99), showfliers=False, patch_artist=True, boxprops={"facecolor": POL_COL[p], "alpha": 0.6})
  ax[0].set_ylabel("|measured joint velocity| [rad/s]")
  ax[1].set_ylabel("|command rate| [rad/s]")
  ax[1].axhline(COMMAND_DELTA_MAX / 0.05, color="k", ls=":", lw=0.8)
  ax[1].set_xticks(range(16), HW_JOINTS, rotation=45)
  from matplotlib.patches import Patch
  ax[0].legend([Patch(color=POL_COL[p], alpha=0.6) for p in pols], pols, ncol=len(pols), fontsize=8)
  fig.suptitle("Joint and command velocity per joint, all attempts pooled (whiskers 1-99%; dotted: runtime max 1/24 rad per 50 ms)")
  fig.tight_layout()
  fig.savefig(fig_dir / "fig_velocity_per_policy.png", dpi=110)
  plt.close(fig)

  # 6: timing
  fig, ax = plt.subplots(1, 2, figsize=(14, 4))
  for p in pols:
    d = np.concatenate([np.diff(r.f["elapsed_s"]) for r in runs if att[r.name]["policy"] == p and r.n > 1]) * 1000
    lc = np.concatenate([r.f["loop_compute_s"] for r in runs if att[r.name]["policy"] == p and r.f["loop_compute_s"] is not None]) * 1000
    ax[0].hist(d, bins=np.arange(40, 120, 1), histtype="step", color=POL_COL[p], label=p, density=True)
    ax[1].hist(lc, bins=np.arange(0, 40, 0.5), histtype="step", color=POL_COL[p], label=p, density=True)
  ax[0].set_yscale("log")
  ax[0].axvline(60, color="k", ls=":")
  ax[0].set(xlabel="control-loop interval [ms]", ylabel="density (log)", title="Loop interval (elapsed_s differences)")
  ax[1].set(xlabel="loop compute time (read + policy + write) [ms]", title="loop_compute_s")
  ax[0].legend()
  fig.tight_layout()
  fig.savefig(fig_dir / "fig_timing.png", dpi=110)
  plt.close(fig)

  # 7: current vs tracking error (quasi-static) with the hardware saturation error
  cve = res["current_vs_error"]
  fig, ax = plt.subplots(1, 4, figsize=(17, 3.8), sharey=True)
  for k, (fn, js) in enumerate(FINGERS.items()):
    for j in js:
      v = cve[HW_JOINTS[j]]
      b = np.array(v["bins_rad"])
      ax[k].plot(0.5 * (b[:-1] + b[1:]), v["median_current_along_error_ma"], "o-", ms=3, label=f"{HW_JOINTS[j]} ({v['slope_ma_per_rad']:.0f} mA/rad)")
    ax[k].axhline(res["config_hand_env"]["current_limit_ma"], color="k", ls=":")
    ax[k].set(title=fn, xlabel="|command - measured| [rad]")
    ax[k].legend(fontsize=7)
  ax[0].set_ylabel("median current along the error [mA]")
  fig.suptitle("Quasi-static (|qd| < 0.15 rad/s) current vs position error, all 37 attempts: a proportional controller clipped at the 350 mA limit")
  fig.tight_layout()
  fig.savefig(fig_dir / "fig_current_vs_tracking_error.png", dpi=110)
  plt.close(fig)

  # 8: lag hardware vs simulation
  lag = res["lag_summary"]
  if lag:
    fig, ax = plt.subplots(figsize=(15, 4))
    xs = np.arange(16)
    for i, p in enumerate([p for p in pols if p in lag]):
      med = [lag[p]["per_joint"][nm]["pos_s"]["median"] * 1000 for nm in HW_JOINTS]
      ax.plot(xs, med, "o-", color=POL_COL[p], label=f"hardware {p} (n={lag[p]['n_runs']} runs >= 10 s)")
    for key, v in res["sim_reference"].items():
      if "lag" in v and "_cap" not in key:  # default (training) actuator settings only
        ax.plot(xs, [v["lag"][nm]["pos_s_median"] * 1000 for nm in HW_JOINTS], "s--", color=POL_COL.get(v["policy"], "k"),
                alpha=0.6, label=f"simulation {v['policy']} ({v['lag'][HW_JOINTS[0]]['n_envs']} envs)")
    ax.set_xticks(xs, HW_JOINTS, rotation=45)
    ax.set_ylabel("effective response lag [ms]")
    ax.set_title("Effective lag: tau minimising |q(t) - command(t - tau)| (unsaturated samples); same estimator for hardware and simulation")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_lag_hardware_vs_sim.png", dpi=110)
    plt.close(fig)

  # 9: guard-stop events
  ev_dir = fig_dir / "guard_events"
  ev_dir.mkdir(exist_ok=True)
  for e in res["guard_events"]:
    r = next(x for x in runs if x.name == e["attempt"])
    tr, _ = times(r)
    I = np.abs(r.f["current_ma"])
    er = np.abs(tracking_errors(r.f["q_hw"], r.f["command_hw"]))
    w = tr >= max(0.0, tr[-1] - 6.0)
    fin = e["finger_highest_current_last_sample"]
    js = FINGERS[fin]
    fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for j in range(16):
      hl = j in js or j == HW_JOINTS.index(e["joint_highest_mean_current_last_1s"])
      ax[0].plot(tr[w], I[w, j], lw=1.4 if hl else 0.5, alpha=1 if hl else 0.4, label=HW_JOINTS[j] if hl else None)
      ax[1].plot(tr[w], er[w, j], lw=1.4 if hl else 0.5, alpha=1 if hl else 0.4, label=HW_JOINTS[j] if hl else None)
    ax[0].axhline(res["config_hand_env"]["current_limit_ma"], color="k", ls=":", lw=0.8)
    ax[0].axhline(res["config_hand_env"]["current_abort_ma"], color="r", ls=":", lw=0.8)
    ax[1].axhline(res["config_hand_env"]["tracking_error_abort_rad"], color="r", ls=":", lw=0.8)
    for i, j in enumerate(js):
      ax[2].plot(tr[w], r.f["command_hw"][w, j], "--", lw=1, color=f"C{i}")
      ax[2].plot(tr[w], r.f["q_hw"][w, j], "-", lw=1.2, color=f"C{i}", label=HW_JOINTS[j])
    ax[3].plot(tr[1:][w[1:]], np.diff(r.f["elapsed_s"])[w[1:]] * 1000, ".-", lw=0.6)
    ax[3].axhline(60, color="k", ls=":")
    for a in ax:
      a.axvline(e["estimated_trip_t_s"], color="r", lw=1.2)
    ax[0].set_ylabel("|current| [mA]")
    ax[1].set_ylabel("|cmd(k-1) - q(k)| [rad]")
    ax[2].set_ylabel(f"{fin}: q (solid), command (dashed) [rad]")
    ax[3].set_ylabel("loop interval [ms]")
    ax[3].set_xlabel("time since policy start [s] (red line: estimated trip; the tripping sample itself is not logged)")
    ax[0].legend(fontsize=7, ncol=5)
    ax[2].legend(fontsize=7, ncol=4)
    tv = e["trip_value"]
    fig.suptitle(f"{e['policy']} {e['attempt']}: {e['stop_reason']} ({e['category']})" +
                 (f"; trip {tv:g}" if tv is not None and math.isfinite(tv) else ""), fontsize=10)
    fig.tight_layout()
    fig.savefig(ev_dir / f"guard_{e['policy']}_{Path(e['attempt']).stem}.png", dpi=100)
    plt.close(fig)


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--date", default="20261120")
  ap.add_argument("--logs", type=Path, default=KIT / "logs")
  ap.add_argument("--csv", type=Path, default=KIT / "logs" / "trials.csv")
  ap.add_argument("--out", type=Path, default=KIT / "audit" / "20261120")
  ap.add_argument("--sim-dir", type=Path, default=None)
  ap.add_argument("--no-figures", action="store_true")
  a = ap.parse_args()
  res = audit(a.date, a.logs, a.csv, a.out, a.sim_dir, figures=not a.no_figures)
  print(summary_markdown(res))
  print(f"-> {a.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
