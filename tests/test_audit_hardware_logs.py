"""Offline hardware-log audit (LEAP_Hand_RL_Deploy_Kit/kit/audit_hardware_logs.py). No hardware.

Synthetic logs in the exact deploy_direct.py format exercise loading, validation, classification,
CSV matching, quarter-turn conversion and the metrics; the last tests run on the real 20 Nov logs
when they are present (read-only)."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "LEAP_Hand_RL_Deploy_Kit"
sys.path.insert(0, str(KIT / "kit"))
sys.path.insert(0, str(KIT / "kit" / "fake_dynamixel"))  # deploy_direct imports dynamixel_sdk

import audit_hardware_logs as A  # noqa: E402

FIELDS = ("trial", "time", "policy", "policy_dir", "checkpoint_sha256", "log", "run_s", "completed", "stop_reason",
          "observed_turns", "dropped", "drop_time_s", "intervened", "notes", "est_turns", "max_current_ma", "max_temp_c",
          "jitter_frac_gt_60ms")


def make_log(path: Path, n: int = 400, *, policy: str = "C5", stop: str = "max_steps", completed: bool = True,
             drop: tuple[str, ...] = (), dt: float = 0.05, current: np.ndarray | None = None, seed: int = 0) -> Path:
  """A log in deploy_direct.py's format: q follows the command one step later (+ small noise)."""
  rng = np.random.default_rng(seed)
  t = 10.0 + np.arange(n) * dt + 0.004
  cmd = 0.5 + 0.3 * np.sin(np.arange(n)[:, None] * 0.1 + np.arange(16)[None] * 0.3)
  q = np.empty_like(cmd)
  q[0] = cmd[0]
  q[1:] = cmd[:-1] + rng.normal(0, 0.003, (n - 1, 16))
  I = current if current is not None else 1500.0 * np.vstack([np.zeros(16), cmd[:-1] - q[1:]])
  temp = np.full((n, 16), np.nan)
  temp[::20] = 30.0
  q, cmd = q.astype(np.float32), cmd.astype(np.float32)  # as logged (float32), so the scalar matches exactly
  te = np.zeros(n)
  te[1:] = np.max(np.abs(q[1:] - cmd[:-1]), axis=1)
  data = {"started_at": np.array("2026-11-20T10:00:00+04:00"), "completed": np.array(completed),
          "stop_reason": np.array(stop), "elapsed_s": t, "loop_compute_s": np.full(n, 0.004), "q_hw": q.astype(np.float32),
          "qd_hw": np.gradient(q, t, axis=0).astype(np.float32), "current_ma": I.astype(np.float32),
          "temperature_c": temp.astype(np.float32), "command_hw": cmd.astype(np.float32), "tracking_error_rad": te,
          "stall_recovery_count": np.zeros(n, np.int32), "stall_recovery_joint": np.full(n, -1, np.int16),
          "run_meta_json": np.array(json.dumps({"policy_dir": f"/x/policies/{policy}", "policy_sha256": f"sha_{policy}"}))}
  for k in drop:
    data.pop(k)
  np.savez_compressed(path, **data)
  return path


def write_csv(path: Path, rows: list[dict]) -> Path:
  with open(path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()
    for r in rows:
      w.writerow({k: r.get(k, "") for k in FIELDS})
  return path


def row(log: str, policy: str = "C5", **kw) -> dict:
  base = {"trial": 1, "policy": policy, "checkpoint_sha256": f"sha_{policy}", "log": log, "run_s": "", "completed": "True",
          "stop_reason": "max_steps", "observed_turns": "", "dropped": "False", "drop_time_s": "", "intervened": "False", "notes": ""}
  base.update(kw)
  return base


# ------------------------------------------------------------------------------- mapping
def test_joint_mapping_matches_deploy_kit():
  import deploy_direct as D

  assert np.array_equal(A.HW_TO_SERVER, D._HW_TO_SERVER)
  assert np.array_equal(A.SERVER_TO_HW, D._SERVER_TO_HW)
  assert np.allclose(A.HW_LIMIT_LO, D._HW_LIMIT_LO) and np.allclose(A.HW_LIMIT_HI, D._HW_LIMIT_HI)
  assert D._JOINTS_PER_DIGIT == 4 and all(js == tuple(range(4 * i, 4 * i + 4)) for i, js in enumerate(A.FINGERS.values()))
  assert A.SERVER_JOINTS[:4] == ("if_mcp", "if_rot", "if_pip", "if_dip")  # training / server order
  x = np.arange(16.0)
  assert np.array_equal(D._server_to_hw(D._hw_to_server(x)), x)
  assert np.array_equal(x[A.HW_TO_SERVER][A.SERVER_TO_HW], x)


def test_hand_env_thresholds_parsed():
  env = A.read_hand_env()
  assert float(env["CURRENT_LIMIT_MA"]) == 350 and float(env["CURRENT_ABORT_MA"]) == 450
  assert float(env["TRACKING_ERROR_ABORT_RAD"]) == 0.65 and env["TRIAL_STEPS"] == "1200"


# ---------------------------------------------------------------------- dimensions / missing
def test_missing_fields_are_flagged_not_zeroed(tmp_path):
  r = A.load_run(make_log(tmp_path / "physical_20990101_000001.npz", drop=("current_ma", "run_meta_json")))
  assert "missing:current_ma" in r.flags and "missing:run_meta_json" in r.flags
  assert r.f["current_ma"] is None and not r.ok
  cl = A.classify(r, None)
  assert cl["category"] == 6
  assert A.analyse_run(r, {}) == {}


def test_wrong_shape_is_flagged(tmp_path):
  p = make_log(tmp_path / "physical_20990101_000002.npz")
  d = dict(np.load(p))
  d["q_hw"] = d["q_hw"][:, :15]
  np.savez(p, **d)
  r = A.load_run(p)
  assert any(f.startswith("shape:q_hw") for f in r.flags) and r.f["q_hw"] is None and not r.ok


def test_truncated_file_is_handled(tmp_path):
  p = make_log(tmp_path / "physical_20990101_000003.npz")
  p.write_bytes(p.read_bytes()[:200])
  r = A.load_run(p)
  assert r.n == 0 and any(f.startswith("unreadable") for f in r.flags)
  assert A.classify(r, None)["category"] == 6


# ---------------------------------------------------------------------------- timestamps
def test_invalid_timestamps_flagged_and_excluded(tmp_path):
  p = make_log(tmp_path / "physical_20990101_000004.npz", n=300)
  d = dict(np.load(p))
  t = d["elapsed_s"].copy()
  t[100] = t[99]  # repeated timestamp
  t[200] = t[199] - 0.01  # goes backwards
  d["elapsed_s"] = t
  np.savez(p, **d)
  r = A.load_run(p)
  assert "timestamps_nonmonotonic" in r.flags
  m = A.analyse_run(r, {})
  assert m["timing"]["n_invalid_intervals"] == 2 and m["timing"]["n_intervals"] == 297
  assert m["lag"] == {}  # lag needs monotonic time

  d["elapsed_s"][5] = np.nan
  np.savez(p, **d)
  r = A.load_run(p)
  assert "timestamps_nonfinite" in r.flags and A.times(r) == (None, None)


def test_irregular_intervals_use_real_timestamps(tmp_path):
  r = A.load_run(make_log(tmp_path / "physical_20990101_000005.npz", n=200))
  t = r.f["elapsed_s"].copy()
  t[50:] += 0.040  # one 90 ms interval
  r.f["elapsed_s"] = t
  m = A.analyse_run(r, {})
  assert m["timing"]["interval_max_ms"] == pytest.approx(90.0, abs=0.5)
  assert m["timing"]["frac_gt_60ms"] == pytest.approx(1 / 199)
  assert m["timing"]["frac_gt_100ms"] == 0.0


# ------------------------------------------------------------------------ classification
@pytest.mark.parametrize("stop,completed,csvrow,expect,flag", [
  ("max_steps", True, {}, 1, None),
  ("max_steps", True, {"dropped": "True", "drop_time_s": "45"}, 2, "ran_to_end_after_drop"),
  ("interrupted", False, {"dropped": "True", "drop_time_s": "5"}, 2, "operator_stopped_after_drop"),
  ("interrupted", False, None, 5, "not_in_trial_record"),
  ("interrupted", False, {"notes": "finger caught on cable"}, 4, None),
  ("interrupted", False, {}, 5, "recorded_without_reason"),
  ("error: Current guard tripped: 462.0 mA", False, None, 3, None),
  ("error: Tracking-error guard tripped: 0.680 rad", False, None, 3, None),
  ("error", False, None, 3, "stop_message_not_logged_pre_patch"),
  ("running", False, None, 6, "chunk_still_running"),
])
def test_classification(tmp_path, stop, completed, csvrow, expect, flag):
  p = make_log(tmp_path / "physical_20990101_000006.npz", n=200, stop=stop, completed=completed)
  r = A.load_run(p)
  rw = None if csvrow is None else row(p.name, **csvrow)
  cl = A.classify(r, rw)
  assert cl["category"] == expect and cl["category_name"] == A.CATEGORIES[expect]
  if flag:
    assert any(f.startswith(flag) for f in cl["flags"])


def test_guard_after_drop_vs_coincident(tmp_path):
  p = make_log(tmp_path / "physical_20990101_000007.npz", n=316, stop="error: Current guard tripped: 462.0 mA", completed=False)
  r = A.load_run(p)  # ~15.75 s
  early = A.classify(r, row(p.name, policy="P0", dropped="True", drop_time_s="5"))
  assert early["category"] == 2 and early["guard_type"] == "current" and early["guard_value"] == 462.0
  same = A.classify(r, row(p.name, policy="P0", dropped="True", drop_time_s="15"))
  assert same["category"] == 3 and same["confidence"] == "low"


def test_guard_parsing():
  assert A.guard_info("error: Current guard tripped: 747.0 mA") == ("current", 747.0)
  assert A.guard_info("error: Tracking-error guard tripped: 0.680 rad") == ("tracking_error", 0.68)
  assert A.guard_info("error: Command-step guard tripped: joint=3, delta=0.0600 rad")[0] == "command_step"
  assert A.guard_info("error")[0] == "unrecorded"
  assert A.guard_info("max_steps") == (None, None) or A.guard_info("max_steps")[0] is None


def test_unrecorded_guard_inferred_from_tracking(tmp_path):
  p = make_log(tmp_path / "physical_20990101_000008.npz", n=100, stop="error", completed=False)
  d = dict(np.load(p))
  d["tracking_error_rad"][-1] = 0.637
  np.savez(p, **d)
  cl = A.classify(A.load_run(p), None, tracking_abort_rad=0.65)
  assert cl["category"] == 3 and cl["inferred_guard"].startswith("tracking_error")


# ------------------------------------------------------------------------- CSV matching
def test_csv_matching_and_reconciliation(tmp_path):
  logs = tmp_path / "logs"
  logs.mkdir()
  a = make_log(logs / "physical_20990101_100000.npz", policy="A3", seed=1)
  b = make_log(logs / "physical_20990101_100100.npz", policy="C5", stop="error: Current guard tripped: 500.0 mA", completed=False, seed=2)
  c = make_log(logs / "physical_20990101_100200.npz", policy="C5", stop="interrupted", completed=False, seed=3)
  write_csv(logs / "trials.csv", [row(a.name, "A3", trial=1, observed_turns="3", run_s="19.95"),
                                  row("physical_20990101_999999.npz", "C5", trial=2)])
  res = A.audit("20990101", logs, logs / "trials.csv", tmp_path / "out", None, figures=False)
  rec = res["reconciliation"]
  assert rec["n_files"] == 3 and rec["n_files_with_csv_row"] == 1 and rec["n_files_without_csv_row"] == 2
  assert rec["csv_rows_without_file"] == ["physical_20990101_999999.npz"]
  assert rec["sums_to_files"] and rec["categories_sum"] == 3
  cats = {x["file"]: x["category"] for x in res["attempts"]}
  assert cats == {a.name: 1, b.name: 3, c.name: 5}
  pols = {x["file"]: x["policy"] for x in res["attempts"]}
  assert pols == {a.name: "A3", b.name: "C5", c.name: "C5"}
  for f in ("attempts.csv", "per_joint.csv", "per_finger.csv", "policy_summary.csv", "guard_events.csv", "audit.json", "summary.md"):
    assert (tmp_path / "out" / f).exists()
  with open(tmp_path / "out" / "per_joint.csv") as fh:
    assert sum(1 for _ in fh) == 1 + 3 * 16


def test_policy_identified_from_metadata_not_order(tmp_path):
  p = make_log(tmp_path / "physical_20990101_000009.npz", policy="R1")
  r = A.load_run(p)
  pol, notes = A.identify_policy(r, row(p.name, "C5"), {"R1": {"sha_R1"}, "C5": {"sha_C5"}})
  assert pol == "R1" and any(n.startswith("csv_policy_mismatch") for n in notes)


# ------------------------------------------------------------------------ quarter turns
def test_quarter_turn_conversion():
  q = A.quarter_turns({"observed_turns": "14", "dropped": "False", "drop_time_s": ""}, 60.0)
  assert q["observed_quarter_turns"] == 14 and q["full_revolutions"] == 3.5 and q["full_rev_per_min_held"] == pytest.approx(3.5)
  d = A.quarter_turns({"observed_turns": "2", "dropped": "True", "drop_time_s": "7"}, 21.6)
  assert d["full_revolutions"] == 0.5 and d["held_s"] == 7 and math.isnan(d["full_rev_per_min_held"])  # held < 20 s
  e = A.quarter_turns({"observed_turns": "11", "dropped": "True", "drop_time_s": "45"}, 60.4)
  assert e["full_rev_per_min_held"] == pytest.approx(11 / 4 / 45 * 60)
  blank = A.quarter_turns({"observed_turns": "", "dropped": "False", "drop_time_s": ""}, 60.0)
  assert math.isnan(blank["observed_quarter_turns"]) and math.isnan(blank["full_revolutions"])
  assert A.quarter_turns(None, 60.0)["observed_quarter_turns"] is None


# --------------------------------------------------------------------- current / tracking
def test_current_metrics_known_signal():
  n = 200
  t = np.arange(n) * 0.05
  I = np.zeros((n, 16))
  I[:, 1] = 100.0
  I[50:70, 1] = 345.0  # 20 samples = 1.0 s at the limit
  I[120, 1] = -420.0   # signed: |current| is used
  m = A.current_metrics(I, t, 0.05)["per_joint"]["if_mcp"]
  assert m["max"] == 420.0 and m["peak_step"] == 120 and m["peak_t_s"] == pytest.approx(6.0)
  assert m["n_ge_340"] == 21 and m["longest_ge_340_s"] == pytest.approx(1.0) and m["episodes_ge_340_over_0.5s"] == 1
  assert m["median"] == 100.0 and m["p99"] > 340
  f = A.finger_metrics(I)
  assert f["index"]["top_joint"] == "if_mcp" and f["index"]["share_of_total"] == pytest.approx(1.0)


def test_tracking_error_reproduces_logged_scalar(tmp_path):
  r = A.load_run(make_log(tmp_path / "physical_20990101_000010.npz"))
  e = A.tracking_errors(r.f["q_hw"], r.f["command_hw"])
  assert np.all(np.isnan(e[0]))  # previous command of step 0 is not logged
  assert np.allclose(np.max(np.abs(e[1:]), axis=1), r.f["tracking_error_rad"][1:])
  m = A.analyse_run(r, {})
  assert m["tracking_logged_reproduced_max_abs_diff"] < 1e-6


def test_lag_estimator_recovers_known_delay():
  dt = 0.05
  t = np.arange(600) * dt
  c = 0.5 * np.sin(2 * np.pi * 0.4 * t) + 0.2 * np.sin(2 * np.pi * 1.1 * t + 0.5)
  for true in (0.05, 0.08, 0.15):
    q = np.interp(t - true, t, c)
    m = np.ones_like(t, bool)
    assert A.lag_position(t, q, t, c, m)[0] == pytest.approx(true, abs=0.006)
    assert A.lag_velocity(t, np.gradient(q, t), t, c, m)[0] == pytest.approx(true, abs=0.011)


def test_current_vs_error_slope_and_saturation(tmp_path):
  logs = []
  for s in range(3):
    p = make_log(tmp_path / f"physical_20990101_00002{s}.npz", n=500, seed=s)
    r = A.load_run(p)
    e = A.tracking_errors(r.f["q_hw"], r.f["command_hw"])
    r.f["current_ma"] = np.clip(np.nan_to_num(1500.0 * e), -350, 350)
    r.f["qd_hw"] = np.zeros_like(r.f["qd_hw"])
    logs.append(r)
  cve = A.current_vs_error(logs)
  assert cve["if_pip"]["slope_ma_per_rad"] == pytest.approx(1500.0, rel=0.02)


# --------------------------------------------------------------- real 20 Nov logs (read-only)
REAL = sorted((KIT / "logs").glob("physical_20261120_*.npz"))


@pytest.mark.skipif(len(REAL) != 37, reason="20 Nov logs not present")
def test_real_logs_reconcile(tmp_path):
  before = {p: p.stat().st_mtime_ns for p in REAL}
  res = A.audit("20261120", KIT / "logs", KIT / "logs" / "trials.csv", tmp_path / "out", None, figures=False)
  assert {p: p.stat().st_mtime_ns for p in REAL} == before  # raw logs untouched
  rec = res["reconciliation"]
  assert rec["n_files"] == 37 and rec["n_files_with_csv_row"] == 24 and rec["sums_to_files"]
  ps = res["policy_summary"]
  assert sum(v["attempts"] for v in ps.values()) == 37
  for p, v in ps.items():
    assert sum(v["categories"].values()) == v["attempts"], p  # mutually exclusive
  c5 = ps["C5"]["categories"]
  assert (ps["C5"]["attempts"], c5["completed_60s"], c5["dropped"], c5["guard_stop"]) == (11, 4, 1, 6)
  assert ps["C5"]["quarter_turns_per_trial_mean"] == pytest.approx(13.9) and ps["C5"]["full_revolutions_per_trial_mean"] == pytest.approx(3.475)
  assert rec["guard_events_total"] == 12
  for a in res["attempts"]:
    assert a["metrics"]["tracking_logged_reproduced_max_abs_diff"] < 1e-6


# ------------------------------------------------------------------------- jam episodes
def test_jam_episodes_resolved_and_unresolved():
  n, dt = 400, 0.05
  t = np.arange(n) * dt
  b = np.zeros((n, 16), bool)
  b[40:70, 1] = True              # 1.5 s jam on if_mcp, resolves
  b[60:66, 2] = True
  b[100:106, 13] = True           # 0.3 s flicker on th_axl: below SUSTAINED_S, dropped
  b[340:, 2] = True               # jam on if_pip until the end of the log
  b[350, 2] = False               # brief dip inside the jam is tolerated
  eps = A.jam_episodes(b, t, dt)
  assert len(eps) == 2
  assert not eps[0]["unresolved"] and eps[0]["duration_s"] == pytest.approx(1.5, abs=0.3) and "if_mcp" in eps[0]["joints"]
  assert eps[1]["unresolved"] and eps[1]["joints"] == ["if_pip"]


def test_trip_classes(tmp_path):
  # sustained jam: index pinned at the limit and far from its command for the last 3 s
  n = 160
  p = make_log(tmp_path / "physical_20990101_000030.npz", n=n, stop="error: Current guard tripped: 700.0 mA", completed=False)
  d = dict(np.load(p))
  d["current_ma"][100:, 1] = 350.0
  d["q_hw"][100:, 1] = d["command_hw"][99:-1, 1] - 0.3
  np.savez(p, **d)
  r = A.load_run(p)
  att = {"policy": "C5", "category_name": "guard_stop", "guard_type": "current", "guard_value": 700.0,
         "metrics": A.analyse_run(r, {})}
  e = A.guard_event(r, att, {})
  assert e["trip_class"] == "sustained_jam" and e["trip_jump_from_last_logged_ma"] == pytest.approx(350.0)
  assert e["joint_highest_mean_current_last_1s"] == "if_mcp" and e["finger_highest_current_last_sample"] == "index"
  # start transient: trips after 1 s with low logged currents
  p2 = make_log(tmp_path / "physical_20990101_000031.npz", n=20, stop="error: Current guard tripped: 800.0 mA", completed=False)
  r2 = A.load_run(p2)
  att2 = {"policy": "P0", "category_name": "guard_stop", "guard_type": "current", "guard_value": 800.0,
          "metrics": A.analyse_run(r2, {})}
  assert A.guard_event(r2, att2, {})["trip_class"] == "start_transient"
