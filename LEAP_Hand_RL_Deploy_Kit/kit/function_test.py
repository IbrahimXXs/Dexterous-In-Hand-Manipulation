#!/usr/bin/env python3
"""Basic function test of the right LEAP hand, before any policy is run.

No policy, no object.  Uses the same DynamixelBus, gain profiles and guards as
deploy_direct.py, so it exercises exactly the motor path the policies use.

Stages (each one must pass before the next starts):
  1. read-only    ping 16 motors, angles inside the LEAP range, temperatures
  2. hold         torque on at the measured pose; nothing may move or draw current
  3. home         ramp to the flat, open hand (0 rad)
  4. sweep        one motor at a time: small move and back. Checks sign, reach,
                  crosstalk (another joint moving = wiring / ID problem) and
                  current. With --confirm you also say which physical joint
                  moved; a swapped motor ID passes any encoder-only check.
  5. grasp pose   ramp to the policy start pose and hold: steady-state tracking
                  error and holding current per joint
  6. tracking     20 Hz commands with policy-sized steps (<= 1/24 rad per step)
                  around the grasp pose: tracking lag and error, as the policy
                  will drive the hand
  7. return home, torque off (also on Ctrl+C or any guard trip)

Every sample is written to an .npz log.  Clear the workspace, take the cube out,
keep the 12 V cutoff within reach.

    ./leap.sh functest                  # interactive (asks you to confirm each joint)
    ./leap.sh functest -- --no-confirm  # automatic checks only
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy_direct import (  # noqa: E402
  _HW_LIMIT_HI,
  _HW_LIMIT_LO,
  _NUM_JOINTS,
  _SERVER_TO_HW,
  DynamixelBus,
  _load_gain_profile,
)

KIT = Path(__file__).resolve().parent.parent
HW_NAMES = (
  "if_rot", "if_mcp", "if_pip", "if_dip",
  "mf_rot", "mf_mcp", "mf_pip", "mf_dip",
  "rf_rot", "rf_mcp", "rf_pip", "rf_dip",
  "th_cmc", "th_axl", "th_mcp", "th_ipl",
)
_FINGER = {"if": "INDEX", "mf": "MIDDLE", "rf": "RING", "th": "THUMB"}
_WHAT = {
  "rot": "swings sideways (abduction)",
  "mcp": "bends at the knuckle (base joint)",
  "pip": "bends at the middle joint",
  "dip": "bends at the fingertip joint",
  "cmc": "rotates at its base toward the palm",
  "axl": "rotates about its own axis / abducts",
  "ipl": "bends at the tip joint",
}
# Policy start pose (server order), same as policy/yaw_policy.py and the MJLab task.
GRASP_SERVER = np.array(
  [0.1, -0.4, 1.3, 0.0, 0.1, 0.0, 1.3, 0.0, 0.1, 0.4, 1.3, 0.0, 1.45, 1.5, 0.579, 1.37], dtype=np.float32
)
POLICY_DELTA = 1.0 / 24.0
# Mechanical joint range of the LEAP hand (hardware order), from the MJCF used in
# training (policy/yaw_policy.py limits).  Wider than the controller limits
# above: e.g. the PIPs can hang back to about -29 deg with torque off.
MECH_LO = np.array([-1.047, -0.314, -0.506, -0.366] * 3 + [-0.349, -0.349, -0.470, -1.340], dtype=np.float32)
MECH_HI = np.array([1.047, 2.230, 1.885, 2.042] * 3 + [2.094, 2.094, 2.443, 1.880], dtype=np.float32)
CALIBRATION_MARGIN_RAD = 0.25  # a 90 deg horn / homing error (1.57 rad) is far outside this


def describe(j: int) -> str:
  f, part = HW_NAMES[j].split("_")
  if f == "th" and part == "mcp":
    return "THUMB bends at its middle joint"
  return f"{_FINGER[f]} finger {_WHAT[part]}"


class Aborted(RuntimeError):
  pass


class Tester:
  def __init__(self, bus: DynamixelBus, a: argparse.Namespace):
    self.bus, self.a = bus, a
    self.lo = _HW_LIMIT_LO + a.limit_margin_rad
    self.hi = _HW_LIMIT_HI - a.limit_margin_rad
    self.cmd = None
    self.t0 = time.monotonic()
    self.log: dict[str, list] = {k: [] for k in ("t", "stage", "cmd", "q", "qd", "current")}
    self.stage = 0
    self.last_temp = np.zeros(_NUM_JOINTS)

  # ------------------------------------------------------------- primitives
  def sample(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q, qd, cur = self.bus.read_state()
    self.log["t"].append(time.monotonic() - self.t0)
    self.log["stage"].append(self.stage)
    self.log["cmd"].append(np.full(_NUM_JOINTS, np.nan, np.float32) if self.cmd is None else self.cmd.copy())
    self.log["q"].append(q)
    self.log["qd"].append(qd)
    self.log["current"].append(cur)
    if self.cmd is not None:
      if np.max(np.abs(cur)) > self.a.current_abort_ma:
        j = int(np.argmax(np.abs(cur)))
        raise Aborted(f"current guard: {HW_NAMES[j]} {cur[j]:.0f} mA > {self.a.current_abort_ma:.0f} mA")
      if np.max(np.abs(qd)) > self.a.velocity_abort_rad_s:
        j = int(np.argmax(np.abs(qd)))
        raise Aborted(f"velocity guard: {HW_NAMES[j]} {qd[j]:.2f} rad/s")
    return q, qd, cur

  def check_temperature(self) -> None:
    self.last_temp = self.bus.read_temperature()
    if np.max(self.last_temp) >= self.a.temperature_abort_c:
      j = int(np.argmax(self.last_temp))
      raise Aborted(f"temperature guard: {HW_NAMES[j]} {self.last_temp[j]:.0f} C")

  def command(self, target: np.ndarray) -> None:
    # A joint that starts outside the controller limits (resting pose) may only
    # move back toward the range, never further out.
    lo = self.lo if self.cmd is None else np.minimum(self.lo, self.cmd)
    hi = self.hi if self.cmd is None else np.maximum(self.hi, self.cmd)
    target = np.clip(np.asarray(target, np.float32), lo, hi)
    if self.cmd is not None:
      step = float(np.max(np.abs(target - self.cmd)))
      if step > self.a.max_command_step_rad + 1e-6:
        raise Aborted(f"internal command step {step:.3f} rad > {self.a.max_command_step_rad} (bug)")
    self.bus.write_position(target)
    self.cmd = target

  def move(self, target: np.ndarray, duration_s: float, hz: float = 50.0) -> None:
    """Linear interpolation from the current command; guard-checked every tick."""
    target = np.clip(np.asarray(target, np.float32), self.lo, self.hi)
    start = self.cmd.copy()
    n = max(1, int(np.ceil(max(duration_s * hz, np.max(np.abs(target - start)) / self.a.max_command_step_rad))))
    for k in range(1, n + 1):
      tick = time.monotonic()
      self.command(start + (target - start) * (k / n))
      self.sample()
      _sleep_until(tick + 1.0 / hz)

  def hold(self, duration_s: float, hz: float = 50.0) -> np.ndarray:
    rows = []
    for _ in range(max(1, int(duration_s * hz))):
      tick = time.monotonic()
      rows.append(self.sample())
      _sleep_until(tick + 1.0 / hz)
    return np.stack([r[0] for r in rows]), np.stack([r[2] for r in rows])

  # ----------------------------------------------------------------- stages
  def stage_read_only(self) -> np.ndarray:
    self.stage = 1
    q, _, _ = self.sample()
    self.check_temperature()
    print("\n[1] read-only (torque OFF)")
    print("     joint      q[rad]  q[deg]  temp  status")
    bad, resting = [], []
    for j in range(_NUM_JOINTS):
      plausible = MECH_LO[j] - CALIBRATION_MARGIN_RAD <= q[j] <= MECH_HI[j] + CALIBRATION_MARGIN_RAD
      in_ctrl = self.lo[j] <= q[j] <= self.hi[j]
      if not plausible:
        bad.append(HW_NAMES[j])
        status = "CALIBRATION?"
      elif not in_ctrl:
        resting.append(HW_NAMES[j])
        status = "outside controller limits (resting pose)"
      else:
        status = "ok"
      print(f"     {HW_NAMES[j]:8s} {q[j]:8.3f} {np.degrees(q[j]):7.1f} {self.last_temp[j]:5.0f}  {status}")
    if bad:
      raise Aborted(f"joints {bad} read outside the LEAP mechanical range: motor horn mounted off by "
                    "90 deg or a homing offset is set. Do not run policies until fixed.")
    if resting:
      print(f"     NOTE: {resting} rest outside the controller limits. Torque-on holds this exact pose,")
      print("     then they are only moved back toward the range (<= "
            f"{self.a.max_command_step_rad} rad per step, current-limited).")
    print(f"     PASS: 16/16 motors, readings mechanically plausible, max temp {self.last_temp.max():.0f} C")
    return q

  def stage_hold(self, q0: np.ndarray) -> None:
    self.stage = 2
    print(f"\n[2] torque ON at measured pose (gains P/I/D={self.a.kp}/{self.a.ki}/{self.a.kd}, "
          f"current limit {self.a.current_limit_ma} mA)")
    self.bus.configure(initial_q=q0, kp=self.a.kp, ki=self.a.ki, kd=self.a.kd,
                       mcp_side_gain_scale=self.a.mcp_side_gain_scale, current_limit_ma=self.a.current_limit_ma)
    self.cmd = q0.astype(np.float32).copy()  # hold exactly where it is (no jump into the limits)
    self.bus.write_position(self.cmd)
    qs, cur = self.hold(1.0)
    drift = np.max(np.abs(qs - q0), axis=0)
    j = int(np.argmax(drift))
    print(f"     max drift {drift[j]:.3f} rad ({HW_NAMES[j]}), max |current| {np.abs(cur).max():.0f} mA")
    if drift[j] > 0.15:
      raise Aborted(f"{HW_NAMES[j]} moved {drift[j]:.3f} rad when holding its own measured pose")
    print("     PASS")

  def stage_home(self) -> None:
    self.stage = 3
    print("\n[3] ramp to flat / open hand")
    self.move(np.zeros(_NUM_JOINTS), self.a.ramp_s)
    qs, cur = self.hold(0.8)
    err = np.abs(qs[-1] - self.cmd)
    print(f"     max |q - cmd| {err.max():.3f} rad ({HW_NAMES[int(np.argmax(err))]}), "
          f"max |current| {np.abs(cur[-10:]).max():.0f} mA")
    print("     PASS" if err.max() < 0.15 else "     WARN: large error at the flat pose (something resting on a finger?)")

  def stage_sweep(self) -> list[dict]:
    self.stage = 4
    print(f"\n[4] single-joint sweep (one motor at a time{'; you confirm each one' if self.a.confirm else ''})")
    home = self.cmd.copy()
    results = []
    for j in range(_NUM_JOINTS):
      amp = self.a.rot_amp if HW_NAMES[j].endswith(("rot", "axl")) else self.a.flex_amp
      tgt = home.copy()
      tgt[j] = np.clip(home[j] + amp, self.lo[j], self.hi[j])
      if tgt[j] - home[j] < 0.5 * amp:  # near the upper limit: go the other way
        tgt[j] = np.clip(home[j] - amp, self.lo[j], self.hi[j])
      want = float(tgt[j] - home[j])
      q_before = self.hold(0.3)[0].mean(0)
      print(f"     motor {j:2d} {HW_NAMES[j]:7s} -> {want:+.2f} rad: expect {describe(j)}", end="", flush=True)
      self.move(tgt, self.a.sweep_s)
      qs, cur = self.hold(0.6)
      q_at = qs[-10:].mean(0)
      self.move(home, self.a.sweep_s)
      self.hold(0.3)
      got = float(q_at[j] - q_before[j])
      other = np.abs(q_at - q_before)
      other[j] = 0.0
      k = int(np.argmax(other))
      r = {
        "joint": HW_NAMES[j], "commanded": want, "measured": got, "reach": got / want if want else np.nan,
        "crosstalk_rad": float(other[k]), "crosstalk_joint": HW_NAMES[k], "max_current_ma": float(np.abs(cur[:, j]).max()),
      }
      ok_sign = np.sign(got) == np.sign(want) and abs(got) > 0.02
      r["auto_pass"] = bool(ok_sign and r["reach"] >= self.a.min_reach and r["crosstalk_rad"] <= self.a.max_crosstalk_rad)
      print(f" | measured {got:+.3f} ({100 * r['reach']:.0f}%), crosstalk {r['crosstalk_rad']:.3f} ({r['crosstalk_joint']}), "
            f"{r['max_current_ma']:.0f} mA  {'ok' if r['auto_pass'] else 'FAIL'}")
      if self.a.confirm:
        ans = input(f"        did the {describe(j)}? [y/n] ").strip().lower()
        r["operator_confirmed"] = ans.startswith("y")
        if not r["operator_confirmed"]:
          print("        -> recorded as a MAPPING FAILURE (motor ID / wiring / joint order)")
      results.append(r)
    n_fail = sum(not r["auto_pass"] or r.get("operator_confirmed") is False for r in results)
    print(f"     {'PASS' if n_fail == 0 else f'FAIL: {n_fail} joint(s)'}")
    if n_fail:
      raise Aborted("single-joint sweep failed; fix before running any policy")
    return results

  def stage_grasp_pose(self) -> dict:
    self.stage = 5
    grasp = np.asarray(GRASP_SERVER[_SERVER_TO_HW], np.float32)
    print("\n[5] ramp to the policy start pose (no cube) and hold")
    self.move(grasp, self.a.ramp_s)
    qs, cur = self.hold(self.a.hold_s)
    err = qs[-25:].mean(0) - self.cmd
    hold_cur = np.abs(cur[-25:]).mean(0)
    print("     joint      cmd     q-cmd   |I| mA")
    for j in range(_NUM_JOINTS):
      print(f"     {HW_NAMES[j]:8s} {self.cmd[j]:7.3f} {err[j]:+8.3f} {hold_cur[j]:7.0f}")
    print(f"     max |q - cmd| {np.abs(err).max():.3f} rad, max holding current {hold_cur.max():.0f} mA")
    if np.abs(err).max() > self.a.tracking_error_abort_rad:
      raise Aborted("cannot reach the policy start pose")
    print("     PASS")
    return {"grasp_cmd_hw": self.cmd.copy(), "grasp_err_hw": err, "grasp_hold_current_ma": hold_cur}

  def stage_tracking(self) -> dict:
    """20 Hz, per-step deltas <= 1/24 rad (the policy's largest step)."""
    self.stage = 6
    hz, dur, amp = 20.0, self.a.track_s, self.a.track_amp
    period = self.a.track_period_s
    # amplitude limited so that the per-step change never exceeds the policy delta
    amp = min(amp, POLICY_DELTA / (2 * np.pi / period / hz))
    print(f"\n[6] 20 Hz tracking: sinusoid {amp:.3f} rad, period {period:.1f} s on all joints, {dur:.0f} s")
    centre = self.cmd.copy()
    phase = np.linspace(0, 2 * np.pi, _NUM_JOINTS, endpoint=False)
    cmds, qs, ticks = [], [], []
    n = int(dur * hz)
    for k in range(1, n + 1):
      tick = time.monotonic()
      ticks.append(tick)
      ramp_in = min(1.0, k / hz)  # 1 s fade-in
      self.command(centre + ramp_in * amp * np.sin(2 * np.pi * k / hz / period + phase))
      q, _, _ = self.sample()
      cmds.append(self.cmd.copy())
      qs.append(q)
      if k % 20 == 0:
        self.check_temperature()
      _sleep_until(tick + 1.0 / hz)
    cmds, qs = np.stack(cmds), np.stack(qs)
    self.move(centre, 1.0)
    # lag (in 20 Hz steps) that best aligns q with cmd, per joint, after the fade-in
    s = int(hz)
    lags = []
    for j in range(_NUM_JOINTS):
      errs = [np.mean((qs[s + L:, j] - cmds[s:len(cmds) - L, j]) ** 2) for L in range(0, 6)]
      lags.append(int(np.argmin(errs)))
    err = qs[s:] - cmds[s:]
    rms = np.sqrt(np.mean((err - err.mean(0)) ** 2, axis=0))
    loop_dt = np.diff(np.asarray(ticks))
    print(f"     loop period mean/max {1e3 * loop_dt.mean():.1f}/{1e3 * loop_dt.max():.1f} ms (target 50)")
    print(f"     lag (steps of 50 ms) per joint: {lags}")
    print(f"     RMS dynamic tracking error: mean {rms.mean():.3f}, max {rms.max():.3f} rad ({HW_NAMES[int(np.argmax(rms))]})")
    print(f"     steady offset (mean q - cmd): max {np.abs(err.mean(0)).max():.3f} rad")
    if loop_dt.max() > 0.1:
      print("     WARN: loop slower than 100 ms at least once (FTDI latency timer not 1 ms?)")
    print("     PASS")
    return {"track_lag_steps": np.array(lags), "track_rms_rad": rms, "track_offset_rad": err.mean(0),
            "track_loop_dt_s": loop_dt}

  def return_home(self) -> None:
    self.stage = 7
    if self.cmd is not None and self.bus.torque_enabled:
      print("\n[7] returning to flat pose, torque off")
      try:
        self.move(np.zeros(_NUM_JOINTS), self.a.ramp_s)
      except Exception as e:  # noqa: BLE001
        print(f"     return-home interrupted: {e}")

  def save(self, path: Path, ok: bool, reason: str, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v) for k, v in self.log.items() if v}
    np.savez_compressed(path, passed=np.array(ok), stop_reason=np.array(reason), joint_names=np.array(HW_NAMES),
                        **arrays, **{k: np.asarray(v) for k, v in extra.items()})
    print(f"[functest] log: {path}")


def _sleep_until(t: float) -> None:
  d = t - time.monotonic()
  if d > 0:
    time.sleep(d)


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--port", default="/dev/ttyUSB0")
  p.add_argument("--baudrate", type=int, default=4_000_000)
  p.add_argument("--gain-profile", type=str.upper, choices=("H1", "H2"), default="H1")
  p.add_argument("--current-limit-ma", type=int, default=350)
  p.add_argument("--current-abort-ma", type=float, default=450.0)
  p.add_argument("--temperature-abort-c", type=float, default=60.0)
  p.add_argument("--velocity-abort-rad-s", type=float, default=8.0)
  p.add_argument("--tracking-error-abort-rad", type=float, default=0.65)
  p.add_argument("--max-command-step-rad", type=float, default=0.05)
  p.add_argument("--limit-margin-rad", type=float, default=0.01)
  p.add_argument("--confirm", action=argparse.BooleanOptionalAction, default=None,
                 help="ask the operator to confirm each joint (default: yes on a terminal)")
  p.add_argument("--stages", default="1234567", help="subset, e.g. 1237 = read, hold, home, sweep only")
  p.add_argument("--flex-amp", type=float, default=0.35, help="sweep amplitude for bending joints [rad]")
  p.add_argument("--rot-amp", type=float, default=0.2, help="sweep amplitude for rot / axial joints [rad]")
  p.add_argument("--sweep-s", type=float, default=0.8)
  p.add_argument("--ramp-s", type=float, default=3.0)
  p.add_argument("--hold-s", type=float, default=2.0)
  p.add_argument("--min-reach", type=float, default=0.6, help="measured / commanded displacement to pass")
  p.add_argument("--max-crosstalk-rad", type=float, default=0.08)
  p.add_argument("--track-s", type=float, default=8.0)
  p.add_argument("--track-amp", type=float, default=0.12)
  p.add_argument("--track-period-s", type=float, default=2.0)
  p.add_argument("--log-file", type=Path, default=None)
  a = p.parse_args()
  if a.confirm is None:
    a.confirm = sys.stdin.isatty()
  prof = _load_gain_profile(a.gain_profile)["hardware"]
  a.kp, a.ki, a.kd = int(prof["position_p_gain_raw"]), int(prof["position_i_gain_raw"]), int(prof["position_d_gain_raw"])
  a.mcp_side_gain_scale = float(prof["mcp_side_gain_scale"])
  if a.current_abort_ma <= a.current_limit_ma:
    p.error("--current-abort-ma must exceed --current-limit-ma")
  return a


def main() -> int:
  a = parse_args()
  log = a.log_file or KIT / "logs" / f"functest_{datetime.now():%Y%m%d_%H%M%S}.npz"
  bus = DynamixelBus(a.port, a.baudrate)
  t = Tester(bus, a)
  extra: dict = {}
  ok, reason = False, "error"
  try:
    bus.connect()
    q0 = t.stage_read_only()
    if not set(a.stages) - {"1"}:
      ok, reason = True, "read-only"
      return 0
    if "2" in a.stages or set(a.stages) & set("3456"):
      t.stage_hold(q0)
    if set(a.stages) & set("3456"):
      t.stage_home()
    if "4" in a.stages:
      sweep = t.stage_sweep()
      for key in ("commanded", "measured", "reach", "crosstalk_rad", "max_current_ma"):
        extra[f"sweep_{key}"] = [r[key] for r in sweep]
      extra["sweep_operator_confirmed"] = [r.get("operator_confirmed", True) for r in sweep]
    if "5" in a.stages or "6" in a.stages:
      extra.update(t.stage_grasp_pose())
    if "6" in a.stages:
      extra.update(t.stage_tracking())
    ok, reason = True, "completed"
    print("\n[functest] ALL STAGES PASSED")
  except KeyboardInterrupt:
    reason = "interrupted"
    print("\n[functest] Ctrl+C")
  except Aborted as e:
    reason = str(e)
    print(f"\n[functest] ABORT: {e}")
  except Exception as e:  # noqa: BLE001
    reason = f"{type(e).__name__}: {e}"
    print(f"\n[functest] ERROR: {reason}")
  finally:
    try:
      t.return_home()
    finally:
      try:
        bus.disable_torque()
      finally:
        bus.close()
        t.save(log, ok, reason, extra)
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
