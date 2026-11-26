#!/usr/bin/env python3
"""C5-HW-V2 step A: actuator response fit (FIT_PLAN.md section A; evaluation only, no training).

record  : frozen C5 seed 1 rollouts on the 24-point grid (delay x damping x friction-loss scale), training
          effort limit (primary) and the 350 mA-equivalent limit (sensitivity).  Recorder = c5_hw_v1's
          ``sim_actuator_reference.record`` imported unchanged; only the three event ranges (and, for the
          sensitivity run, the effort range) are replaced in the loaded task config.  Several workers can share
          the grid (claim files).
score   : the audit's ``sim_metrics`` on every rollout, the pre-registered selection, stop rule and the
          randomisation ranges around the fit -> results/response_fit.json + .md
functest: free-air cross-check, hardware vs simulation (functest_20261117_094855.npz, stage 6)

    python experiments/c5_hw_v2/scripts/response_fit.py record [--effort default|cap350]
    python experiments/c5_hw_v2/scripts/response_fit.py score
    python experiments/c5_hw_v2/scripts/response_fit.py functest
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
REPO = PROJECT.parents[1]
E = PROJECT / "experiments" / "c5_hw_v2"
KIT = REPO / "LEAP_Hand_RL_Deploy_Kit"
DATA = E / "data" / "response_fit"
SEED, N_ENVS, STEPS = 8401, 256, 600
DELAYS_MS = {"0-25": (0, 5), "0-50": (0, 10), "25-50": (5, 10), "25-75": (5, 15)}  # physics steps of 5 ms
DAMPING = (0.5, 0.75, 1.0)
FRICTION = (0.5, 1.0)
BASE_DAMPING, BASE_FRICTION = (0.6, 1.6), (0.5, 1.8)
EFFORT = {"default": None, "cap350": (0.16695, 0.20405)}
TRACK_BAND, STOP_S, JOINT_OK_S = (0.70, 1.30), 0.150, 0.030


def _load(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[name] = mod  # dataclasses need the module registered
  spec.loader.exec_module(mod)
  return mod


def audit():
  return _load("audit_hardware_logs", KIT / "kit" / "audit_hardware_logs.py")


def grid():
  for d, sd, sf in itertools.product(DELAYS_MS, DAMPING, FRICTION):
    yield {"delay": d, "damping": sd, "friction": sf, "name": f"d{d}_D{sd:g}_F{sf:g}"}


@contextlib.contextmanager
def patched_events(delay=None, damping=None, friction=None, effort=None, extra=None):
  """Replace event ranges in every task config loaded inside the block (the task files are not touched)."""
  import mjlab.tasks.registry as reg

  base = reg.load_env_cfg

  def patched(task, *a, **k):
    cfg = base(task, *a, **k)
    if delay is not None:
      cfg.events["dr_action_delay"].params["lag_range"] = tuple(int(x) for x in delay)
    if damping is not None:
      cfg.events["dr_motor_damping"].params["ranges"] = tuple(float(x) for x in damping)
    if friction is not None:
      cfg.events["dr_motor_friction"].params["ranges"] = tuple(float(x) for x in friction)
    if effort is not None:
      cfg.events["dr_effort_limits"].params["effort_limit"] = tuple(float(x) for x in effort)
    if extra is not None:
      extra(cfg)
    return cfg

  reg.load_env_cfg = patched
  try:
    yield
  finally:
    reg.load_env_cfg = base


def point_ranges(p: dict) -> dict:
  return {"delay": DELAYS_MS[p["delay"]], "damping": tuple(p["damping"] * x for x in BASE_DAMPING),
          "friction": tuple(p["friction"] * x for x in BASE_FRICTION)}


def cmd_record(a) -> None:
  ref = _load("sim_actuator_reference", E.parent / "c5_hw_v1" / "scripts" / "sim_actuator_reference.py")
  DATA.mkdir(parents=True, exist_ok=True)
  (DATA / ".claims").mkdir(exist_ok=True)
  eff = EFFORT[a.effort]
  for p in grid():
    out = DATA / f"{a.effort}_{p['name']}.npz"
    claim = DATA / ".claims" / out.stem
    if out.exists():
      continue
    if claim.exists() and time.time() - claim.stat().st_mtime > 3600:
      claim.unlink(missing_ok=True)
    try:
      os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
      continue
    t0 = time.time()
    rg = point_ranges(p)
    with patched_events(**rg):
      r = ref.record("C5", SEED, N_ENVS, STEPS, a.device, eff)  # eff=None -> task's training range
    r["grid_point"] = np.array(json.dumps({**p, **{k: list(v) for k, v in rg.items()}, "effort": a.effort}))
    np.savez_compressed(out, **r)
    print(f"[{out.stem}] saved ({time.time() - t0:.0f}s), alive at end {r['alive'][-1].mean():.3f}", flush=True)


def hw_targets(A) -> dict:
  lag = json.loads((KIT / "audit" / "20261120" / "lag_summary.json").read_text())["C5"]["per_joint"]
  lag_hw = {j: float(lag[j]["pos_s"]["median"]) for j in A.HW_JOINTS}
  per = {}
  with open(KIT / "audit" / "20261120" / "per_joint.csv", newline="") as f:
    for r in csv.DictReader(f):
      if r["policy"] == "C5" and r["category_name"] == "completed_60s":
        per.setdefault(r["joint"], []).append(float(r["track_median"]))
  track_hw = {j: float(np.median(per[j])) for j in A.HW_JOINTS}
  n_runs = {j: len(per[j]) for j in A.HW_JOINTS}
  sat = json.loads((KIT / "audit" / "20261120" / "current_vs_error.json").read_text())
  hw_sat = {}
  for nm, v in sat.items():
    s = v.get("slope_ma_per_rad") if isinstance(v, dict) else None
    if s and np.isfinite(s) and s > 0:
      hw_sat[nm] = 350.0 / s
  return {"lag_hw": lag_hw, "track_hw": track_hw, "track_hw_summary": float(np.median(list(track_hw.values()))),
          "track_hw_runs_per_joint": n_runs, "hw_sat_err": hw_sat}


def score_file(A, path: Path, tgt: dict) -> dict:
  m = A.sim_metrics(path, tgt["hw_sat_err"], n_envs_lag=32)
  lag = {j: m["lag"][j]["pos_s_median"] for j in A.HW_JOINTS}
  lagv = {j: m["lag"][j]["vel_fd_s_median"] for j in A.HW_JOINTS}
  trk = {j: m["tracking"][j]["median"] for j in A.HW_JOINTS}
  d = {j: abs(lag[j] - tgt["lag_hw"][j]) for j in A.HW_JOINTS}
  ts = float(np.median(list(trk.values())))
  return {"lag_pos_s": lag, "lag_vel_s": lagv, "tracking_median": trk, "abs_lag_err_s": d,
          "sum_abs_lag_err_s": float(sum(d.values())), "median_abs_lag_err_s": float(np.median(list(d.values()))),
          "median_lag_s": float(np.median(list(lag.values()))), "tracking_summary": ts,
          "tracking_ratio": ts / tgt["track_hw_summary"],
          "joints_within_30ms": int(sum(v <= JOINT_OK_S + 1e-9 for v in d.values())),
          "alive_frac_end": m["alive_frac_end"], "n_envs_lag": m["lag"][A.HW_JOINTS[0]]["n_envs"],
          "abs_qfd_mean_all": m["abs_qfd_mean_all"]}


def select(rows: dict, tgt: dict) -> dict:
  ok = {k: r for k, r in rows.items() if TRACK_BAND[0] <= r["tracking_ratio"] <= TRACK_BAND[1]}
  best_lag = min(rows, key=lambda k: rows[k]["sum_abs_lag_err_s"])
  if not ok:
    return {"status": "failed_tracking", "selected": None, "best_lag_point_for_information": best_lag,
            "n_points_meeting_tracking_constraint": 0}
  sel = min(ok, key=lambda k: ok[k]["sum_abs_lag_err_s"])
  r = rows[sel]
  status = "ok" if r["median_abs_lag_err_s"] <= STOP_S else "failed_structural"
  return {"status": status, "selected": sel, "best_lag_point_for_information": best_lag,
          "n_points_meeting_tracking_constraint": len(ok)}


def randomisation(p: dict) -> dict:
  lo, hi = DELAYS_MS[p["delay"]]
  c_ms = 2.5 * (lo + hi)  # midpoint in ms (5 ms per physics step)
  d_lo, d_hi = max(0.0, c_ms - 25.0), c_ms + 25.0
  dv, fv = p["damping"] * 1.1, p["friction"] * 1.15
  return {"delay_ms": [d_lo, d_hi], "lag_range_steps": [int(round(d_lo / 5)), int(round(d_hi / 5))],
          "damping_scale_range": [0.8 * dv, 1.25 * dv], "friction_scale_range": [0.8 * fv, 1.25 * fv]}


def cmd_score(a) -> None:
  A = audit()
  tgt = hw_targets(A)
  res = {"plan_sha256": (E / "logs" / "fit_plan.sha256").read_text().split()[0], "targets": tgt, "runs": {}}
  for eff in EFFORT:
    rows = {}
    for p in grid():
      f = DATA / f"{eff}_{p['name']}.npz"
      if not f.exists():
        raise SystemExit(f"missing {f}")
      rows[p["name"]] = {**p, **score_file(A, f, tgt)}
    res["runs"][eff] = {"points": rows, "selection": select(rows, tgt)}
  sel = res["runs"]["default"]["selection"]
  res["primary"] = sel
  if sel["status"] == "ok":
    p = res["runs"]["default"]["points"][sel["selected"]]
    res["fitted_randomisation"] = randomisation(p)
  cur = "d25-75_D1_F1"
  res["current_setting"] = {e: {k: res["runs"][e]["points"][cur][k] for k in (
    "median_lag_s", "median_abs_lag_err_s", "sum_abs_lag_err_s", "tracking_summary", "tracking_ratio", "joints_within_30ms")}
    for e in EFFORT}
  (E / "results").mkdir(exist_ok=True)
  (E / "results" / "response_fit.json").write_text(json.dumps(res, indent=1) + "\n")
  write_md(res)
  print(json.dumps({"primary": sel, "current": res["current_setting"]["default"]}, indent=1))


def write_md(res: dict) -> None:
  t = res["targets"]
  L = ["# C5-HW-V2 step A: response fit (generated by scripts/response_fit.py score)", "",
       f"Plan: FIT_PLAN.md (sha256 {res['plan_sha256'][:16]}...). Frozen C5 seed 1, grasp 49, 256 envs, seed {SEED}, 30 s.",
       f"Hardware targets: C5 median lag over joints {1000 * np.median(list(t['lag_hw'].values())):.0f} ms; "
       f"tracking summary {t['track_hw_summary']:.4f} rad (C5 completed runs). Tracking band "
       f"{TRACK_BAND[0] * t['track_hw_summary']:.4f}-{TRACK_BAND[1] * t['track_hw_summary']:.4f} rad.", ""]
  for eff, run in res["runs"].items():
    L += [f"## Effort limit: {eff}" + (" (primary)" if eff == "default" else " (sensitivity, not used for selection)"), "",
          "| point | median lag (ms) | median abs lag err (ms) | sum abs lag err (s) | joints within 30 ms | tracking (rad) | ratio to hw | alive at 30 s |",
          "|---|---|---|---|---|---|---|---|"]
    for k, r in sorted(run["points"].items(), key=lambda kv: kv[1]["sum_abs_lag_err_s"]):
      L.append(f"| {k} | {1000 * r['median_lag_s']:.0f} | {1000 * r['median_abs_lag_err_s']:.0f} | {r['sum_abs_lag_err_s']:.3f} | "
               f"{r['joints_within_30ms']} | {r['tracking_summary']:.4f} | {r['tracking_ratio']:.2f} | {r['alive_frac_end']:.3f} |")
    s = run["selection"]
    L += ["", f"Selection: **{s['status']}**; selected {s['selected']}; points meeting the tracking band: "
          f"{s['n_points_meeting_tracking_constraint']}; best lag point (information): {s['best_lag_point_for_information']}.", ""]
  if "fitted_randomisation" in res:
    L += ["## Fitted randomisation for C5-HW-V2", "", "```", json.dumps(res["fitted_randomisation"], indent=1), "```", ""]
  if "functest" in res:
    L += ["## Free-air cross-check", "", "```", json.dumps(res["functest"], indent=1), "```"]
  (E / "results" / "response_fit.md").write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------------------------- functest
def functest_hw(A) -> dict:
  d = np.load(KIT / "logs" / "functest_20261117_094855.npz", allow_pickle=False)
  m = d["stage"] == 6
  t, c, q = d["t"][m].astype(float), d["cmd"][m].astype(float), d["q"][m].astype(float)
  t = t - t[0]
  keep = t >= 1.0  # after the 1 s fade-in
  names = [str(x) for x in d["joint_names"]]
  assert tuple(names) == A.HW_JOINTS, names
  lag = {nm: A.lag_position(t, q[:, j], t, c[:, j], keep)[0] for j, nm in enumerate(names)}
  return {"t": t, "cmd": c, "lag": lag, "n": int(m.sum()), "dt_median": float(np.median(np.diff(t)))}


def functest_sim(A, hw: dict, ranges: dict, device: str, n: int = 64) -> dict:
  import torch
  from in_hand_rotation_mjlab.grasp_rotation_v2 import rollout as R

  cmd_hw = hw["cmd"]
  T = len(cmd_hw)
  with patched_events(**ranges):
    env, raw, _ = R.make_env("Mjlab-Leap-RetV1-A", n, SEED, device, num_steps=T + 5,
                             terminations_off=tuple(raw_terms()))
  robot, cube = raw.scene["robot"], raw.scene["cube"]
  term = raw.action_manager.get_term("joint_pos")
  jn = [str(x) for x in robot.joint_names]
  assert tuple(jn) == A.SERVER_JOINTS, jn
  targets = {"x": None}

  def process_actions(actions):
    term._raw_actions[:] = actions
    term._prev_target[:] = term._target
    term._substep_counter = 0
    term._target = targets["x"].clone()
    term._processed_actions = term._target

  term.process_actions = process_actions
  env.reset()
  far = cube.data.root_link_pose_w.clone()
  far[:, 2] -= 0.5  # cube 0.5 m below the hand: no contact (free-air)
  cube.write_root_link_pose_to_sim(far)
  cube.write_root_link_velocity_to_sim(torch.zeros((n, 6), device=device))
  cmd_srv = torch.as_tensor(cmd_hw[:, A.HW_TO_SERVER], dtype=torch.float32, device=device)
  # start at the functest's first command (the policy start pose) and hold 1 s, as the functest does
  q0 = cmd_srv[0].expand(n, -1).clone()
  robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
  targets["x"] = q0
  zero = torch.zeros((n, 16), device=device)
  for _ in range(20):
    env.step(zero)
  qs = np.zeros((T, n, 16))
  for k in range(T):
    targets["x"] = cmd_srv[k].expand(n, -1).clone()
    env.step(zero)
    qs[k] = robot.data.joint_pos.cpu().numpy()
  env.close()
  dt = float(raw.step_dt)
  t = np.arange(T) * dt
  keep = t >= 1.0
  q_hw = qs[..., A.SERVER_TO_HW]  # -> hardware order
  # q read after step k corresponds to the hardware read after command k: compare with the command series
  lag = {}
  for j, nm in enumerate(A.HW_JOINTS):
    v = [A.lag_position(t + dt, q_hw[:, i, j], t, cmd_hw[:, j], keep)[0] for i in range(n)]
    lag[nm] = float(np.nanmedian(v))
  return lag


def raw_terms():
  from in_hand_rotation_mjlab.estimation.sim import _import_tasks
  from mjlab.tasks.registry import load_env_cfg

  _import_tasks()
  return list(load_env_cfg("Mjlab-Leap-RetV1-A").terminations)


def cmd_functest(a) -> None:
  A = audit()
  hw = functest_hw(A)
  res_path = E / "results" / "response_fit.json"
  res = json.loads(res_path.read_text())
  out = {"hardware_lag_s": hw["lag"], "hardware_median_lag_s": float(np.median(list(hw["lag"].values()))),
         "hardware_samples": hw["n"], "hardware_dt_median_s": hw["dt_median"],
         "note": "hardware stage-6 samples are ~50 ms apart (20 Hz), lag grid 5 ms with linear command interpolation"}
  pts = {"current": point_ranges({"delay": "25-75", "damping": 1.0, "friction": 1.0})}
  if res["primary"]["status"] == "ok":
    sel = res["runs"]["default"]["points"][res["primary"]["selected"]]
    pts["selected"] = point_ranges(sel)
  else:  # no selected point: the best-lag point, for information only (labelled as such)
    info = res["runs"]["default"]["points"][res["primary"]["best_lag_point_for_information"]]
    pts["best_lag_point_information_only"] = point_ranges(info)
  for k, rg in pts.items():
    lag = functest_sim(A, hw, rg, a.device)
    out[f"sim_{k}_lag_s"] = lag
    out[f"sim_{k}_median_lag_s"] = float(np.median(list(lag.values())))
  res["functest"] = out
  res_path.write_text(json.dumps(res, indent=1) + "\n")
  write_md(res)
  print(json.dumps({k: v for k, v in out.items() if "median" in k}, indent=1))


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("cmd", choices=("record", "score", "functest"))
  ap.add_argument("--effort", choices=tuple(EFFORT), default="default")
  ap.add_argument("--device", default="cuda:0")
  a = ap.parse_args()
  {"record": cmd_record, "score": cmd_score, "functest": cmd_functest}[a.cmd](a)


if __name__ == "__main__":
  main()
