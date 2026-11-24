"""Read-only robot telemetry: the kit's UDP stream (live) and run logs (.npz, replay).

Packet format (from LEAP_Hand_RL_Deploy_Kit/kit/deploy_direct.py ``LiveStream``, one JSON
datagram per 20 Hz control step, sent only when the run is started with ``--live-udp``)::

  {"type": "step", "step": int, "t": seconds since run start, "shadow": bool, "port": str,
   "policy": str, "q_server": [16], "cmd_server": [16], "current_ma_hw": [16],
   "temp_max_c": float, "monitor": {cube_pos_palm, cube_quat_palm, ..., est_yaw_rad} | null}

There is no wall-clock time in the packet, so live packets are stamped with the receive
time on this host (same host/clock as the camera frames).  Run logs carry ``started_at``
and ``elapsed_s`` (monotonic from that instant): host time = started_at + elapsed_s.

Guard stops are NOT sent over UDP; the log's ``stop_reason`` has them.  Live, the only
observable is that packets stop arriving ("telemetry stopped", reason unknown).

This module only *receives* (and can relay datagrams to another local viewer port); it
never opens the serial port or talks to the motors.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
import datetime as _dt
import json
from pathlib import Path
import socket
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[4]
KIT_DIR = REPO / "LEAP_Hand_RL_Deploy_Kit" / "kit"
SERVER_JOINT_NAMES = ("if_mcp", "if_rot", "if_pip", "if_dip", "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
                      "rf_mcp", "rf_rot", "rf_pip", "rf_dip", "th_cmc", "th_axl", "th_mcp", "th_ipl")
# finger -> hardware motor ids (kit/live_viewer.py FINGER_MOTORS)
FINGER_MOTORS = {"index": [0, 1, 2, 3], "middle": [4, 5, 6, 7], "ring": [8, 9, 10, 11], "thumb": [12, 13, 14, 15]}
HW_TO_SERVER = np.array([1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15])


def kit_import(name: str):
  if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))
  return __import__(name)


def read_hand_env(path: Path = REPO / "LEAP_Hand_RL_Deploy_Kit" / "configs" / "hand.env") -> dict:
  """Read-only parse of the kit's safety settings (displayed; never modified)."""
  out = {}
  try:
    for line in Path(path).read_text().splitlines():
      line = line.split("#", 1)[0].strip()
      if "=" in line:
        k, v = line.split("=", 1)
        v = v.strip()
        if v.startswith("${") and ":-" in v:
          v = v.split(":-", 1)[1].rstrip("}")
        out[k.strip()] = v
  except OSError:
    pass
  return out


@dataclass
class TelemetrySample:
  host_time_s: float
  packet: dict

  @property
  def t_run(self) -> float:
    return float(self.packet.get("t", np.nan))


class TelemetryBuffer:
  """Time-ordered samples with nearest-in-time lookup (explicit staleness)."""

  def __init__(self, max_len: int = 200_000):
    self.times: list[float] = []
    self.samples: list[TelemetrySample] = []
    self.max_len = max_len

  def add(self, s: TelemetrySample) -> None:
    if self.times and s.host_time_s < self.times[-1]:
      i = bisect.bisect(self.times, s.host_time_s)
      self.times.insert(i, s.host_time_s)
      self.samples.insert(i, s)
    else:
      self.times.append(s.host_time_s)
      self.samples.append(s)
    if len(self.times) > self.max_len:
      del self.times[0], self.samples[0]

  def nearest(self, host_time_s: float, max_dt_s: float) -> tuple[TelemetrySample | None, float | None]:
    """Latest sample at or before ``host_time_s`` (causal); (None, age) if older than max_dt_s."""
    if not self.times:
      return None, None
    i = bisect.bisect_right(self.times, host_time_s) - 1
    if i < 0:
      return None, None
    age = host_time_s - self.times[i]
    return (self.samples[i] if age <= max_dt_s else None), age

  def window(self, t0: float, t1: float) -> list[TelemetrySample]:
    i0 = bisect.bisect_left(self.times, t0)
    i1 = bisect.bisect_right(self.times, t1)
    return self.samples[i0:i1]


class UdpTelemetry:
  """Listen for the kit's datagrams.  ``forward``: optionally relay each datagram unchanged to
  another local port (e.g. a kit live_viewer started on 127.0.0.1:47801)."""

  def __init__(self, listen: str, forward: str | None = None):
    host, port = listen.rsplit(":", 1)
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
      self.sock.bind((host, int(port)))
    except OSError as exc:
      raise RuntimeError(f"cannot listen on udp {listen} ({exc}); is the kit live_viewer already bound to it? "
                         "Use a different port and start the run with: ./leap.sh run <policy> -- --live-udp <host:port>") from exc
    self.sock.setblocking(False)
    self.fwd = None
    if forward:
      fh, fp = forward.rsplit(":", 1)
      if (fh, int(fp)) == (host, int(port)):
        raise ValueError("--forward-udp must differ from the listen address")
      self.fwd = ((fh, int(fp)), socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
    self.label = f"udp {listen}" + (f" (relay -> {forward})" if forward else "")
    self.n_rx = 0
    self.n_bad = 0
    self.last_rx_host = None

  def poll(self) -> list[TelemetrySample]:
    out = []
    while True:
      try:
        data, _ = self.sock.recvfrom(65536)
      except BlockingIOError:
        return out
      now = time.time()
      if self.fwd is not None:
        try:
          self.fwd[1].sendto(data, self.fwd[0])
        except OSError:
          pass
      try:
        pk = json.loads(data)
      except ValueError:
        self.n_bad += 1
        continue
      if isinstance(pk, dict) and pk.get("type") == "step":
        self.n_rx += 1
        self.last_rx_host = now
        out.append(TelemetrySample(now, pk))

  def close(self) -> None:
    self.sock.close()


@dataclass
class RunLog:
  path: Path
  samples: list[TelemetrySample]
  started_at: _dt.datetime
  stop_reason: str
  completed: bool
  monitor_source: str
  meta: dict

  @property
  def t0_host(self) -> float:
    return self.samples[0].host_time_s if self.samples else float("nan")


def load_run_log(path: Path, monitor_dir: Path | None = None) -> RunLog:
  """Kit run log -> samples in the UDP packet format (monitor fields re-computed by the kit's
  own code when the log has none)."""
  path = Path(path)
  d = np.load(path, allow_pickle=False)
  started = _dt.datetime.fromisoformat(str(d["started_at"]))
  elapsed = np.asarray(d["elapsed_s"], np.float64)
  meta = json.loads(str(d["run_meta_json"])) if "run_meta_json" in d.files else {}
  lv = kit_import("live_viewer")
  pk, how = lv.load_log_packets(path, Path(monitor_dir) if monitor_dir else KIT_DIR.parent / "monitors")
  samples = []
  for i, p in enumerate(pk):
    p["t_run_s"] = float(elapsed[i])  # seconds since started_at (the viewer's "t" starts at 0)
    if "tracking_error_rad" in d.files:
      p["tracking_error_rad"] = float(d["tracking_error_rad"][i])
    samples.append(TelemetrySample(started.timestamp() + float(elapsed[i]), p))
  return RunLog(path, samples, started, str(d["stop_reason"]), bool(d["completed"]), how, meta)


def finger_currents(packet: dict) -> dict[str, float]:
  cur = np.abs(np.asarray(packet.get("current_ma_hw", np.full(16, np.nan)), np.float64))
  return {f: float(cur[ids].sum()) for f, ids in FINGER_MOTORS.items()}


def joint_currents_server(packet: dict) -> np.ndarray:
  cur = np.asarray(packet.get("current_ma_hw", np.full(16, np.nan)), np.float64)
  return cur[HW_TO_SERVER]


def command_minus_measured(packet: dict) -> np.ndarray:
  """Per joint (server order): applied command - measured position [rad] (a tracking error, not a force)."""
  q = np.asarray(packet.get("q_server", np.full(16, np.nan)), np.float64)
  c = np.asarray(packet.get("cmd_server", np.full(16, np.nan)), np.float64)
  return c - q
