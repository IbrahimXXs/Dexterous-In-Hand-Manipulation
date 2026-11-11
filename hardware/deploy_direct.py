#!/usr/bin/env python3
"""Deploy a server-hosted policy directly over a Dynamixel serial bus.

This is a ROS-free alternative to ``real_world/deploy.py``.  It is intended
for short, supervised experiments and adds hardware-limit clipping, current /
velocity / temperature guards, a safe controller startup order, bounded run
duration, run logging, and return-to-home on successful completion.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
import json
import time
from pathlib import Path
from typing import Any

from dynamixel_sdk import (
  COMM_SUCCESS,
  GroupSyncRead,
  GroupSyncWrite,
  PacketHandler,
  PortHandler,
)
import numpy as np
import zmq


try:
  from .cube_pose_io import CubePosePolicyTerms, JsonCubePoseSource
except ImportError:
  # Support direct execution as: python real_world/deploy_direct.py
  from cube_pose_io import CubePosePolicyTerms, JsonCubePoseSource


_NUM_JOINTS = 16
_MOTOR_IDS = tuple(range(_NUM_JOINTS))
_GAIN_PROFILE_DIR = (
  Path(__file__).resolve().parent.parent
  / "configs"
  / "leap_right_gain_profiles"
)
_PROTOCOL_VERSION = 2.0
_POS_SCALE = 2.0 * np.pi / 4096.0
_VEL_SCALE = 0.229 * 2.0 * np.pi / 60.0

_ADDR_OPERATING_MODE = 11
_ADDR_TORQUE_ENABLE = 64
_ADDR_D_GAIN = 80
_ADDR_I_GAIN = 82
_ADDR_P_GAIN = 84
_ADDR_CURRENT_LIMIT = 102
_ADDR_PROFILE_ACCELERATION = 108
_ADDR_PROFILE_VELOCITY = 112
_ADDR_GOAL_POSITION = 116
_ADDR_PRESENT_CURRENT = 126
_ADDR_PRESENT_VELOCITY = 128
_ADDR_PRESENT_POSITION = 132
_ADDR_PRESENT_TEMPERATURE = 146

# Hardware order:
# [if_rot, if_mcp, if_pip, if_dip, ..., th_cmc, th_axl, th_mcp, th_ipl]
# Server order swaps rot/mcp for the three fingers.
_HW_TO_SERVER = np.array(
  [1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15],
  dtype=np.int64,
)
_SERVER_TO_HW = np.argsort(_HW_TO_SERVER).astype(np.int64)

# Four contiguous hardware-order joints belong to each digit:
# index, middle, ring, and thumb. Stall recovery backs off only the digit
# containing the most persistently loaded joint.
_JOINTS_PER_DIGIT = 4

# Physical right-hand limits from leap_hand_controller.
_HW_LIMIT_LO = np.array(
  [
    -0.523599, -0.261799, -0.261799, -0.261799,
    -0.523599, -0.261799, -0.261799, -0.261799,
    -0.523599, -0.261799, -0.261799, -0.261799,
    0.0, 0.0, -0.349066, -0.349066,
  ],
  dtype=np.float32,
)
_HW_LIMIT_HI = np.array(
  [
    0.523599, 1.91986, 1.91986, 1.91986,
    0.523599, 1.91986, 1.91986, 1.91986,
    0.523599, 1.91986, 1.91986, 1.91986,
    1.571, 1.571, 1.74533, 1.74533,
  ],
  dtype=np.float32,
)


def _load_gain_profile(context: str | None) -> dict[str, Any] | None:
  if context is None:
    return None
  parent = Path(__file__).resolve().parent.parent
  candidate_dirs = [
    parent / "config" / "leap_right_gain_profiles",
    parent / "configs" / "leap_right_gain_profiles",
    _GAIN_PROFILE_DIR,
  ]
  for d in candidate_dirs:
    path = d / f"{context}.json"
    if path.is_file():
      profile = json.loads(path.read_text(encoding="utf-8"))
      if profile.get("context") != context:
        raise ValueError(
          f"Gain profile {path} declares context {profile.get('context')!r}"
        )
      return profile
  raise FileNotFoundError(
    f"Gain profile '{context}.json' not found in any of {candidate_dirs}"
  )



def _signed(value: int, size: int) -> int:
  sign_bit = 1 << (8 * size - 1)
  return value - (1 << (8 * size)) if value & sign_bit else value


def _encode_array(array: np.ndarray) -> dict[str, Any]:
  x = np.ascontiguousarray(array)
  return {
    "dtype": str(x.dtype),
    "shape": list(x.shape),
    "data_b64": base64.b64encode(x.tobytes(order="C")).decode("ascii"),
  }


def _decode_array(payload: dict[str, Any]) -> np.ndarray:
  dtype = np.dtype(str(payload["dtype"]))
  shape = tuple(int(v) for v in payload["shape"])
  raw = base64.b64decode(str(payload["data_b64"]))
  return np.frombuffer(raw, dtype=dtype).copy().reshape(shape)


def _hw_to_server(q: np.ndarray) -> np.ndarray:
  return np.asarray(q, dtype=np.float32)[_HW_TO_SERVER].copy()


def _server_to_hw(q: np.ndarray) -> np.ndarray:
  return np.asarray(q, dtype=np.float32)[_SERVER_TO_HW].copy()


def _observation(
  q_hw: np.ndarray,
  prev_cmd_hw: np.ndarray,
  pose_terms: CubePosePolicyTerms | None = None,
  last_action_server: np.ndarray | None = None,
) -> dict[str, Any]:
  row = lambda x: np.asarray(x[None, :], dtype=np.float32)
  q_server = _hw_to_server(q_hw)
  prev_server = _hw_to_server(prev_cmd_hw)
  if last_action_server is None:
    last_action_server = np.zeros(_NUM_JOINTS, dtype=np.float32)
  term_values = {
    "joint_pos": _encode_array(row(q_server)),
    "prev_commanded_joint_pos": _encode_array(row(prev_server)),
    "joint_pos_error": _encode_array(row(q_server - prev_server)),
    "last_action": _encode_array(row(last_action_server)),
  }
  if pose_terms is not None:
    term_values["camera_cube_pose"] = _encode_array(
      row(pose_terms.camera_cube_pose)
    )
    term_values["pose_goal"] = _encode_array(row(pose_terms.pose_goal))
    term_values["camera_cube_position_error"] = _encode_array(
      row(pose_terms.camera_cube_position_error)
    )
    term_values["camera_cube_orientation_error"] = _encode_array(
      row(pose_terms.camera_cube_orientation_error)
    )
    if pose_terms.translation_goal_error is not None:
      term_values["translation_goal_error"] = _encode_array(
        row(pose_terms.translation_goal_error)
      )
    if pose_terms.pickup_phase is not None:
      term_values["pickup_phase"] = _encode_array(
        row(pose_terms.pickup_phase)
      )
    if pose_terms.pickup_progress is not None:
      term_values["pickup_progress"] = _encode_array(
        row(pose_terms.pickup_progress)
      )
  return {
    "term_values": term_values,
    "action_term_joint_pos": {
      "joint_pos": _encode_array(row(q_server)),
    },
  }


class PolicyClient:
  def __init__(self, endpoint: str, timeout_ms: int):
    self.endpoint = endpoint
    self._is_local = endpoint.lower() in ("local", "embedded", "none")
    if self._is_local:
      sys_root = Path(__file__).resolve().parents[1]
      if str(sys_root) not in sys.path:
        sys.path.insert(0, str(sys_root))
      from policy.yaw_policy import ContinuousYawPolicy
      self._policy = ContinuousYawPolicy()
      self._socket = None
    else:
      self._socket = zmq.Context.instance().socket(zmq.REQ)
      self._socket.setsockopt(zmq.LINGER, 0)
      self._socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
      self._socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
      self._socket.connect(endpoint)

  def request(self, payload: dict[str, Any]) -> dict[str, Any]:
    if self._is_local:
      msg_type = payload.get("type", "")
      if msg_type == "ping":
        return {"ok": True}
      elif msg_type == "init":
        return {
          "ok": True,
          "actor_observation_terms": [
            "joint_pos",
            "prev_commanded_joint_pos",
            "joint_pos_error",
            "last_action",
          ],
          "action_term_metadata": {
            "joint_pos": {
              "default_joint_pos": _encode_array(
                self._policy.default_joint_pos.astype(np.float32)
              ),
            },
          },
        }
      elif msg_type == "reset":
        obs = payload.get("observation", {})
        term_values = obs.get("term_values", {})
        if "joint_pos" in term_values:
          q_server = _decode_array(term_values["joint_pos"]).reshape(-1)
          self._policy.reset(q_server)
        else:
          self._policy.reset()
        return {"ok": True}
      elif msg_type == "infer":
        obs = payload.get("observation", {})
        term_values = obs.get("term_values", {})
        if "joint_pos" not in term_values:
          raise RuntimeError("Missing joint_pos in local policy infer request")
        q_server = _decode_array(term_values["joint_pos"]).reshape(-1)
        new_target, raw_action, _ = self._policy.step(q_server)
        row_target = np.asarray(new_target[None, :], dtype=np.float32)
        row_raw = np.asarray(raw_action[None, :], dtype=np.float32)
        return {
          "ok": True,
          "action": {
            "raw_action": _encode_array(row_raw),
            "joint_position_commands": {
              "joint_pos": _encode_array(row_target),
            },
          },
        }
      elif msg_type == "close":
        return {"ok": True}
      return {"ok": False, "error": f"Unknown local message type {msg_type}"}

    self._socket.send_json(payload)
    response = self._socket.recv_json()
    if not bool(response.get("ok", False)):
      raise RuntimeError(str(response.get("error", "Policy server error")))
    return response

  def close(self) -> None:
    if self._socket is not None:
      self._socket.close(linger=0)



class DynamixelBus:
  def __init__(self, port: str, baudrate: int):
    self.port = PortHandler(port)
    self.packet = PacketHandler(_PROTOCOL_VERSION)
    self.baudrate = int(baudrate)
    self.connected = False
    self.torque_enabled = False

    self.state_reader = GroupSyncRead(
      self.port, self.packet, _ADDR_PRESENT_CURRENT, 10
    )
    self.temperature_reader = GroupSyncRead(
      self.port, self.packet, _ADDR_PRESENT_TEMPERATURE, 1
    )
    for motor_id in _MOTOR_IDS:
      if not self.state_reader.addParam(motor_id):
        raise RuntimeError(f"Could not add motor {motor_id} to state reader")
      if not self.temperature_reader.addParam(motor_id):
        raise RuntimeError(
          f"Could not add motor {motor_id} to temperature reader"
        )

  def connect(self) -> None:
    if not self.port.openPort():
      raise RuntimeError("Could not open Dynamixel serial port")
    if not self.port.setBaudRate(self.baudrate):
      self.port.closePort()
      raise RuntimeError(f"Could not set serial baudrate {self.baudrate}")
    self.connected = True

    missing = []
    for motor_id in _MOTOR_IDS:
      _, comm, error = self.packet.ping(self.port, motor_id)
      if comm != COMM_SUCCESS or error != 0:
        missing.append(motor_id)
    if missing:
      raise RuntimeError(f"Motor ping failed for IDs {missing}")

  def _check_comm(self, comm: int, context: str) -> None:
    if comm != COMM_SUCCESS:
      raise RuntimeError(f"{context}: {self.packet.getTxRxResult(comm)}")

  def _write_register_all(
    self, *, address: int, size: int, values: np.ndarray
  ) -> None:
    writer = GroupSyncWrite(self.port, self.packet, address, size)
    for motor_id, value in zip(_MOTOR_IDS, values, strict=True):
      unsigned = int(value) & ((1 << (8 * size)) - 1)
      if not writer.addParam(
        motor_id, unsigned.to_bytes(size, byteorder="little")
      ):
        raise RuntimeError(
          f"Could not stage register {address} for motor {motor_id}"
        )
    self._check_comm(writer.txPacket(), f"sync-write address {address}")
    writer.clearParam()

  def _write_byte_verified(self, address: int, value: int) -> None:
    failed = []
    for motor_id in _MOTOR_IDS:
      comm, error = self.packet.write1ByteTxRx(
        self.port, motor_id, address, int(value)
      )
      if comm != COMM_SUCCESS or error != 0:
        failed.append(motor_id)
    if failed:
      raise RuntimeError(
        f"Register {address} write failed for motors {failed}"
      )

  def read_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    success = False
    last_comm = None
    for _ in range(10):
      last_comm = self.state_reader.txRxPacket()
      if last_comm == COMM_SUCCESS:
        success = True
        break
      self.port.clearPort()
      time.sleep(0.002)
    if not success:
      assert last_comm is not None
      self._check_comm(last_comm, "state read")

    q = np.empty(_NUM_JOINTS, dtype=np.float32)
    qd = np.empty(_NUM_JOINTS, dtype=np.float32)
    current = np.empty(_NUM_JOINTS, dtype=np.float32)
    for i, motor_id in enumerate(_MOTOR_IDS):
      if not self.state_reader.isAvailable(
        motor_id, _ADDR_PRESENT_CURRENT, 10
      ):
        raise RuntimeError(f"State unavailable for motor {motor_id}")
      raw_current = self.state_reader.getData(
        motor_id, _ADDR_PRESENT_CURRENT, 2
      )
      raw_velocity = self.state_reader.getData(
        motor_id, _ADDR_PRESENT_VELOCITY, 4
      )
      raw_position = self.state_reader.getData(
        motor_id, _ADDR_PRESENT_POSITION, 4
      )
      current[i] = float(_signed(raw_current, 2))
      qd[i] = float(_signed(raw_velocity, 4)) * _VEL_SCALE
      q[i] = float(_signed(raw_position, 4)) * _POS_SCALE - np.pi
    if not (
      np.all(np.isfinite(q))
      and np.all(np.isfinite(qd))
      and np.all(np.isfinite(current))
    ):
      raise RuntimeError("Non-finite Dynamixel state")
    return q, qd, current

  def read_temperature(self) -> np.ndarray:
    success = False
    last_comm = None
    for _ in range(10):
      last_comm = self.temperature_reader.txRxPacket()
      if last_comm == COMM_SUCCESS:
        success = True
        break
      self.port.clearPort()
      time.sleep(0.002)
    if not success:
      assert last_comm is not None
      self._check_comm(last_comm, "temperature read")
    values = np.empty(_NUM_JOINTS, dtype=np.float32)
    for i, motor_id in enumerate(_MOTOR_IDS):
      if not self.temperature_reader.isAvailable(
        motor_id, _ADDR_PRESENT_TEMPERATURE, 1
      ):
        raise RuntimeError(f"Temperature unavailable for motor {motor_id}")
      values[i] = self.temperature_reader.getData(
        motor_id, _ADDR_PRESENT_TEMPERATURE, 1
      )
    return values

  def write_position(self, q_hw: np.ndarray) -> None:
    raw = np.rint((np.asarray(q_hw) + np.pi) / _POS_SCALE).astype(np.int64)
    self._write_register_all(
      address=_ADDR_GOAL_POSITION, size=4, values=raw
    )

  def configure(
    self,
    *,
    initial_q: np.ndarray,
    kp: int,
    ki: int,
    kd: int,
    mcp_side_gain_scale: float,
    current_limit_ma: int,
  ) -> None:
    # Prewrite the measured pose before any torque transition.  This prevents a
    # stale persisted goal from causing a startup snap.
    self.write_position(initial_q)
    self._write_byte_verified(_ADDR_TORQUE_ENABLE, 0)
    self.torque_enabled = False
    self._write_byte_verified(_ADDR_OPERATING_MODE, 5)
    self._write_register_all(
      address=9, size=1, values=np.zeros(_NUM_JOINTS, dtype=np.int64)
    )
    self._write_register_all(
      address=_ADDR_PROFILE_VELOCITY,
      size=4,
      values=np.full(_NUM_JOINTS, 125, dtype=np.int64),
    )
    self._write_register_all(
      address=_ADDR_PROFILE_ACCELERATION,
      size=4,
      values=np.zeros(_NUM_JOINTS, dtype=np.int64),
    )
    p_values = np.full(_NUM_JOINTS, int(kp), dtype=np.int64)
    i_values = np.full(_NUM_JOINTS, int(ki), dtype=np.int64)
    d_values = np.full(_NUM_JOINTS, int(kd), dtype=np.int64)
    mcp_side_ids = [0, 4, 8]
    p_values[mcp_side_ids] = int(round(mcp_side_gain_scale * kp))
    i_values[mcp_side_ids] = int(round(mcp_side_gain_scale * ki))
    d_values[mcp_side_ids] = int(round(mcp_side_gain_scale * kd))
    self._write_register_all(address=_ADDR_P_GAIN, size=2, values=p_values)
    self._write_register_all(
      address=_ADDR_I_GAIN,
      size=2,
      values=i_values,
    )
    self._write_register_all(address=_ADDR_D_GAIN, size=2, values=d_values)
    self._write_register_all(
      address=_ADDR_CURRENT_LIMIT,
      size=2,
      values=np.full(_NUM_JOINTS, current_limit_ma, dtype=np.int64),
    )
    self.write_position(initial_q)
    self._write_byte_verified(_ADDR_TORQUE_ENABLE, 1)
    self.torque_enabled = True

  def disable_torque(self) -> None:
    if self.connected:
      self._write_byte_verified(_ADDR_TORQUE_ENABLE, 0)
    self.torque_enabled = False

  def close(self) -> None:
    if self.connected:
      self.port.closePort()
      self.connected = False


def _ramp(
  bus: DynamixelBus,
  start: np.ndarray,
  target: np.ndarray,
  *,
  duration_s: float,
  hz: float,
  current_abort_ma: float,
  velocity_abort_rad_s: float,
) -> None:
  steps = max(1, int(round(duration_s * hz)))
  period = 1.0 / hz
  for step in range(1, steps + 1):
    loop_start = time.monotonic()
    alpha = step / steps
    command = (1.0 - alpha) * start + alpha * target
    bus.write_position(command)
    if step % 5 == 0 or step == steps:
      _, qd, current = bus.read_state()
      if float(np.max(np.abs(current))) > current_abort_ma:
        raise RuntimeError(
          f"Ramp current guard tripped: {np.max(np.abs(current)):.1f} mA"
        )
      if float(np.max(np.abs(qd))) > velocity_abort_rad_s:
        raise RuntimeError(
          f"Ramp velocity guard tripped: {np.max(np.abs(qd)):.2f} rad/s"
        )
    remaining = period - (time.monotonic() - loop_start)
    if remaining > 0:
      time.sleep(remaining)

def _stall_release_target(
  q_hw: np.ndarray,
  prev_cmd_hw: np.ndarray,
  *,
  stalled_joint: int,
  release_rad: float,
  lower: np.ndarray,
  upper: np.ndarray,
) -> np.ndarray:
  """Unload and slightly back off the digit containing ``stalled_joint``.

  The sign comes from the position error that generated load: if the command
  is ahead of the measured position, move the release target behind the
  measured position (and vice versa). Other digits hold their measured pose.
  """
  if not 0 <= stalled_joint < _NUM_JOINTS:
    raise ValueError(f"Invalid stalled joint index {stalled_joint}")
  if release_rad <= 0.0:
    raise ValueError("release_rad must be positive")

  q_hw = np.asarray(q_hw, dtype=np.float32)
  prev_cmd_hw = np.asarray(prev_cmd_hw, dtype=np.float32)
  target = q_hw.copy()
  first = (stalled_joint // _JOINTS_PER_DIGIT) * _JOINTS_PER_DIGIT
  digit_ids = np.arange(first, first + _JOINTS_PER_DIGIT)
  error = prev_cmd_hw[digit_ids] - q_hw[digit_ids]
  active = np.abs(error) > 1e-4
  target[digit_ids[active]] -= np.sign(error[active]) * release_rad
  return np.clip(target, lower, upper)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--server-endpoint", default="tcp://127.0.0.1:5555")
  parser.add_argument("--server-timeout-ms", type=int, default=30_000)
  parser.add_argument("--port", default="/dev/ttyUSB0")
  parser.add_argument("--baudrate", type=int, default=4_000_000)
  parser.add_argument("--policy-hz", type=float, default=20.0)
  parser.add_argument(
    "--pose-input-file",
    type=Path,
    default=None,
    help=(
      "Atomically updated camera/goal JSON packet. Required when the server "
      "policy declares camera_cube_pose or pose_goal observations."
    ),
  )
  parser.add_argument(
    "--max-pose-age-s",
    type=float,
    default=0.25,
    help="Abort if the camera pose packet is older than this many seconds.",
  )
  parser.add_argument(
    "--shadow",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Run live inference and guards without enabling torque or writing goals.",
  )
  parser.add_argument(
    "--max-steps",
    type=int,
    default=120,
    help="Number of policy steps; 0 runs until interrupted.",
  )
  parser.add_argument(
    "--start-delay-s",
    type=float,
    default=0.0,
    help="Supervised cube-placement delay before torque is enabled.",
  )
  parser.add_argument("--init-duration-s", type=float, default=5.0)
  parser.add_argument("--return-home-duration-s", type=float, default=3.0)
  parser.add_argument(
    "--gain-profile",
    type=str.upper,
    choices=("H1", "H2"),
    default=None,
    help=(
      "Load hardware P/I/D from configs/leap_right_gain_profiles; overrides "
      "--kp, --ki, --kd, and --mcp-side-gain-scale."
    ),
  )
  parser.add_argument("--kp", type=int, default=600)
  parser.add_argument("--ki", type=int, default=0)
  parser.add_argument("--kd", type=int, default=150)
  parser.add_argument("--mcp-side-gain-scale", type=float, default=0.75)
  parser.add_argument("--current-limit-ma", type=int, default=450)
  parser.add_argument("--current-abort-ma", type=float, default=700.0)
  parser.add_argument("--velocity-abort-rad-s", type=float, default=8.0)
  parser.add_argument("--temperature-abort-c", type=float, default=65.0)
  parser.add_argument("--tracking-error-abort-rad", type=float, default=0.65)
  parser.add_argument("--max-command-step-rad", type=float, default=0.05)
  parser.add_argument("--limit-margin-rad", type=float, default=0.01)
  parser.add_argument("--temperature-interval-steps", type=int, default=20)
  parser.add_argument("--status-interval-steps", type=int, default=200)
  parser.add_argument(
    "--stall-recovery",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
      "Recover from sustained contact stalls by unloading and slightly "
      "backing off the affected digit, then resetting policy history/targets."
    ),
  )
  parser.add_argument("--stall-current-ma", type=float, default=300.0)
  parser.add_argument("--stall-tracking-error-rad", type=float, default=0.15)
  parser.add_argument("--stall-hold-steps", type=int, default=5)
  parser.add_argument("--stall-release-rad", type=float, default=0.12)
  parser.add_argument("--stall-release-duration-s", type=float, default=0.5)
  parser.add_argument("--stall-settle-s", type=float, default=0.25)
  parser.add_argument("--stall-cooldown-steps", type=int, default=40)
  parser.add_argument("--max-stall-recoveries", type=int, default=10)
  parser.add_argument(
    "--log-chunk-steps",
    type=int,
    default=1200,
    help="Flush this many samples per NPZ chunk; 0 buffers until exit.",
  )
  parser.add_argument(
    "--return-home-on-interrupt",
    action=argparse.BooleanOptionalAction,
    default=True,
  )
  parser.add_argument("--log-file", type=Path, default=None)
  return parser.parse_args()


def _save_records(
  *,
  path: Path,
  records: list[dict[str, Any]],
  started_at: datetime,
  completed: bool,
  stop_reason: str,
) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    path,
    started_at=np.array(started_at.isoformat()),
    completed=np.array(completed),
    stop_reason=np.array(stop_reason),
    elapsed_s=np.asarray([r["elapsed_s"] for r in records]),
    loop_compute_s=np.asarray([r["loop_compute_s"] for r in records]),
    q_hw=np.stack([r["q_hw"] for r in records]),
    qd_hw=np.stack([r["qd_hw"] for r in records]),
    current_ma=np.stack([r["current_ma"] for r in records]),
    temperature_c=np.stack([r["temperature_c"] for r in records]),
    command_hw=np.stack([r["command_hw"] for r in records]),
    camera_cube_pose=np.stack([r["camera_cube_pose"] for r in records]),
    pose_goal=np.stack([r["pose_goal"] for r in records]),
    translation_goal_error=np.stack(
      [r["translation_goal_error"] for r in records]
    ),
    tracking_error_rad=np.asarray(
      [r["tracking_error_rad"] for r in records]
    ),
    stall_recovery_count=np.asarray(
      [r["stall_recovery_count"] for r in records], dtype=np.int32
    ),
    stall_recovery_joint=np.asarray(
      [r["stall_recovery_joint"] for r in records], dtype=np.int16
    ),
  )
  print(f"[direct] saved run log: {path}")


def main() -> None:
  args = _parse_args()
  gain_profile = _load_gain_profile(args.gain_profile)
  if gain_profile is not None:
    hardware_cfg = gain_profile["hardware"]
    mujoco_cfg = gain_profile["mujoco"]
    args.kp = int(hardware_cfg["position_p_gain_raw"])
    args.ki = int(hardware_cfg["position_i_gain_raw"])
    args.kd = int(hardware_cfg["position_d_gain_raw"])
    args.mcp_side_gain_scale = float(hardware_cfg["mcp_side_gain_scale"])
    print(
      f"[direct] gain profile {args.gain_profile}: hardware P/I/D="
      f"{args.kp}/{args.ki}/{args.kd}, MuJoCo kp/kv="
      f"{mujoco_cfg['kp_Nm_per_rad']}/{mujoco_cfg['kv_Nm_s_per_rad']}"
    )
  if args.max_steps < 0:
    raise ValueError("--max-steps must be non-negative")
  if not all(0 <= gain <= 65535 for gain in (args.kp, args.ki, args.kd)):
    raise ValueError("--kp, --ki, and --kd must fit unsigned 16-bit registers")
  if not 0.0 < args.mcp_side_gain_scale <= 1.0:
    raise ValueError("--mcp-side-gain-scale must be in (0, 1]")
  if args.current_abort_ma <= args.current_limit_ma:
    raise ValueError("--current-abort-ma must exceed --current-limit-ma")
  if args.temperature_interval_steps <= 0:
    raise ValueError("--temperature-interval-steps must be positive")
  if args.status_interval_steps <= 0:
    raise ValueError("--status-interval-steps must be positive")
  if args.log_chunk_steps < 0:
    raise ValueError("--log-chunk-steps must be non-negative")
  if args.max_pose_age_s <= 0.0:
    raise ValueError("--max-pose-age-s must be positive")
  if args.stall_recovery:
    if not 0.0 < args.stall_current_ma < args.current_abort_ma:
      raise ValueError(
        "--stall-current-ma must be positive and below --current-abort-ma"
      )
    if not 0.0 < args.stall_tracking_error_rad < args.tracking_error_abort_rad:
      raise ValueError(
        "--stall-tracking-error-rad must be positive and below "
        "--tracking-error-abort-rad"
      )
    if args.stall_hold_steps <= 0:
      raise ValueError("--stall-hold-steps must be positive")
    if args.stall_release_rad <= 0.0:
      raise ValueError("--stall-release-rad must be positive")
    if args.stall_release_duration_s <= 0.0:
      raise ValueError("--stall-release-duration-s must be positive")
    if args.stall_settle_s < 0.0:
      raise ValueError("--stall-settle-s must be non-negative")
    if args.stall_cooldown_steps < 0:
      raise ValueError("--stall-cooldown-steps must be non-negative")
    if args.max_stall_recoveries <= 0:
      raise ValueError("--max-stall-recoveries must be positive")
  pose_source = (
    None
    if args.pose_input_file is None
    else JsonCubePoseSource(args.pose_input_file, args.max_pose_age_s)
  )

  rpc = PolicyClient(args.server_endpoint, args.server_timeout_ms)
  bus = DynamixelBus(args.port, args.baudrate)
  completed = False
  stopped_cleanly = False
  stop_reason = "error"
  records: list[dict[str, Any]] = []
  saved_log_files: list[Path] = []
  chunk_index = 0
  started_at = datetime.now().astimezone()
  started_monotonic = time.monotonic()
  log_file = args.log_file
  if log_file is None:
    stamp = started_at.strftime("%Y%m%d_%H%M%S")
    log_prefix = "shadow_policy_right" if args.shadow else "physical_policy_right"
    log_file = Path(__file__).resolve().parent.parent / "data" / (
      f"{log_prefix}_{stamp}.npz"
    )

  try:
    rpc.request({"type": "ping"})
    init = rpc.request(
      {
        "type": "init",
        "step_dt": 1.0 / args.policy_hz,
        "max_episode_steps": 10_000_000,
      }
    )
    actor_terms = set(init.get("actor_observation_terms", ()))
    supported_terms = {
      "joint_pos",
      "prev_commanded_joint_pos",
      "joint_pos_error",
      "last_action",
      "camera_cube_pose",
      "pose_goal",
      "camera_cube_position_error",
      "camera_cube_orientation_error",
      "translation_goal_error",
      "pickup_phase",
      "pickup_progress",
    }
    unsupported_terms = actor_terms - supported_terms
    if unsupported_terms:
      raise RuntimeError(
        "This direct client cannot provide server observation terms "
        f"{sorted(unsupported_terms)}."
      )
    pose_terms_required = {
      "camera_cube_pose",
      "pose_goal",
      "camera_cube_position_error",
      "camera_cube_orientation_error",
      "translation_goal_error",
      "pickup_phase",
      "pickup_progress",
    } & actor_terms
    if pose_terms_required and pose_source is None:
      raise RuntimeError(
        "The selected policy requires camera/goal observations "
        f"{sorted(pose_terms_required)}. Supply --pose-input-file."
      )
    if "action_term_metadata" in init and "joint_pos" in init["action_term_metadata"]:
      metadata = init["action_term_metadata"]["joint_pos"]
      default_server = _decode_array(
        metadata["default_joint_pos"]
      ).reshape(-1).astype(np.float32)
    else:
      # Older servers omit action_term_metadata; fall back to the policy's
      # known default grasp pose (server joint order).
      print(
        "[direct] WARNING: server init response missing 'action_term_metadata'; "
        "using built-in default joint positions."
      )
      default_server = np.array(
        [0.1, -0.4, 1.3, 0.0, 0.1, 0.0, 1.3, 0.0,
         0.1,  0.4, 1.3, 0.0, 1.45, 1.5, 0.579, 1.37],
        dtype=np.float32,
      )
    if default_server.shape != (_NUM_JOINTS,):
      raise RuntimeError(f"Unexpected default pose shape {default_server.shape}")
    default_hw = _server_to_hw(default_server)

    lower = _HW_LIMIT_LO + args.limit_margin_rad
    upper = _HW_LIMIT_HI - args.limit_margin_rad
    if np.any(lower >= upper):
      raise ValueError("Joint-limit margin is too large")
    default_hw = np.clip(default_hw, lower, upper)

    bus.connect()
    initial_q, initial_qd, _ = bus.read_state()
    initial_temp = bus.read_temperature()
    if float(np.max(initial_temp)) >= args.temperature_abort_c:
      raise RuntimeError(
        f"Initial temperature guard tripped: {np.max(initial_temp):.1f} C"
      )
    print(
      "[direct] connected; initial q_hw="
      + np.array2string(initial_q, precision=3)
    )
    if args.shadow:
      print(
        f"[direct] temperature range={initial_temp.min():.0f}-"
        f"{initial_temp.max():.0f} C; SHADOW mode, torque remains disabled"
      )
      q_hw = initial_q.copy()
    else:
      print(
        f"[direct] temperature range={initial_temp.min():.0f}-"
        f"{initial_temp.max():.0f} C; configuring conservative Mode 5"
      )
      if args.start_delay_s > 0:
        print(
          f"[direct] place/center the cube now; motion starts in "
          f"{args.start_delay_s:.1f} s"
        )
        time.sleep(args.start_delay_s)

      bus.configure(
        initial_q=initial_q,
        kp=args.kp,
        ki=args.ki,
        kd=args.kd,
        mcp_side_gain_scale=args.mcp_side_gain_scale,
        current_limit_ma=args.current_limit_ma,
      )
      print(
        f"[direct] moving to policy grasp over {args.init_duration_s:.1f} s"
      )
      _ramp(
        bus,
        initial_q,
        default_hw,
        duration_s=args.init_duration_s,
        hz=50.0,
        current_abort_ma=args.current_abort_ma,
        velocity_abort_rad_s=args.velocity_abort_rad_s,
      )
      q_hw, _, _ = bus.read_state()
    # The action mapper is reset from the measured post-ramp pose, so the
    # previous command used by both the observation and step guard must start
    # from that same measured pose.  Contact/compliance can leave it offset
    # from the nominal default pose.
    prev_cmd_hw = q_hw.copy()
    last_action_server = np.zeros(_NUM_JOINTS, dtype=np.float32)
    pose_terms = None if pose_source is None else pose_source.read()
    rpc.request(
      {
        "type": "reset",
        "observation": _observation(
          q_hw, prev_cmd_hw, pose_terms, last_action_server
        ),
      }
    )

    run_length = str(args.max_steps) if args.max_steps else "unbounded"
    print(
      f"[direct] policy loop started: {run_length} steps at "
      f"{args.policy_hz:.1f} Hz"
    )
    period = 1.0 / args.policy_hz
    last_temperature_c = initial_temp.copy()
    stall_counts = np.zeros(_NUM_JOINTS, dtype=np.int32)
    stall_recovery_count = 0
    stall_recovery_joint = -1
    stall_cooldown_steps = 0
    step = 0
    while args.max_steps == 0 or step < args.max_steps:
      loop_start = time.monotonic()
      q_hw, qd_hw, current_ma = bus.read_state()
      if step % args.temperature_interval_steps == 0:
        temperature_c = bus.read_temperature()
      else:
        temperature_c = np.full(_NUM_JOINTS, np.nan, dtype=np.float32)

      if float(np.max(np.abs(current_ma))) > args.current_abort_ma:
        raise RuntimeError(
          f"Current guard tripped: {np.max(np.abs(current_ma)):.1f} mA"
        )
      if float(np.max(np.abs(qd_hw))) > args.velocity_abort_rad_s:
        raise RuntimeError(
          f"Velocity guard tripped: {np.max(np.abs(qd_hw)):.2f} rad/s"
        )
      finite_temp = temperature_c[np.isfinite(temperature_c)]
      if finite_temp.size:
        last_temperature_c = temperature_c.copy()
      if finite_temp.size and float(np.max(finite_temp)) >= args.temperature_abort_c:
        raise RuntimeError(
          f"Temperature guard tripped: {np.max(finite_temp):.1f} C"
        )
      tracking_error_by_joint = np.abs(q_hw - prev_cmd_hw)
      tracking_error = float(np.max(tracking_error_by_joint))
      if not args.shadow and step >= 10 and tracking_error > args.tracking_error_abort_rad:
        raise RuntimeError(
          f"Tracking-error guard tripped: {tracking_error:.3f} rad"
        )

      if args.stall_recovery and not args.shadow:
        if stall_cooldown_steps > 0:
          stall_cooldown_steps -= 1
          stall_counts.fill(0)
        else:
          stalled_now = (
            (np.abs(current_ma) >= args.stall_current_ma)
            & (tracking_error_by_joint >= args.stall_tracking_error_rad)
          )
          stall_counts = np.where(stalled_now, stall_counts + 1, 0)
          if int(np.max(stall_counts)) >= args.stall_hold_steps:
            if stall_recovery_count >= args.max_stall_recoveries:
              raise RuntimeError(
                "Maximum stall recoveries reached: "
                f"{stall_recovery_count}"
              )
            stalled_joint = int(np.argmax(stall_counts))
            digit = stalled_joint // _JOINTS_PER_DIGIT
            print(
              f"[direct] stall detected: step={step} digit={digit} "
              f"joint={stalled_joint} "
              f"current={abs(current_ma[stalled_joint]):.0f} mA "
              f"tracking_error={tracking_error_by_joint[stalled_joint]:.3f} rad; "
              "unloading digit"
            )

            # Commanding the measured pose immediately removes accumulated
            # position error; it does not request an instantaneous movement.
            hold_target = q_hw.copy()
            bus.write_position(hold_target)
            time.sleep(min(0.1, args.stall_release_duration_s))
            release_target = _stall_release_target(
              q_hw,
              prev_cmd_hw,
              stalled_joint=stalled_joint,
              release_rad=args.stall_release_rad,
              lower=lower,
              upper=upper,
            )
            _ramp(
              bus,
              hold_target,
              release_target,
              duration_s=args.stall_release_duration_s,
              hz=50.0,
              current_abort_ma=args.current_abort_ma,
              velocity_abort_rad_s=args.velocity_abort_rad_s,
            )
            if args.stall_settle_s > 0.0:
              time.sleep(args.stall_settle_s)

            q_hw, _, _ = bus.read_state()
            prev_cmd_hw = q_hw.copy()
            last_action_server.fill(0.0)
            pose_terms = None if pose_source is None else pose_source.read()
            rpc.request(
              {
                "type": "reset",
                "observation": _observation(
                  q_hw, prev_cmd_hw, pose_terms, last_action_server
                ),
              }
            )
            stall_recovery_count += 1
            stall_recovery_joint = stalled_joint
            stall_cooldown_steps = args.stall_cooldown_steps
            stall_counts.fill(0)
            print(
              f"[direct] stall recovery {stall_recovery_count}/"
              f"{args.max_stall_recoveries} completed; policy reset"
            )
            continue
      pose_terms = None if pose_source is None else pose_source.read()
      reply = rpc.request(
        {
          "type": "infer",
          "observation": _observation(
            q_hw, prev_cmd_hw, pose_terms, last_action_server
          ),
        }
      )
      raw_action_server = _decode_array(
        reply["action"]["raw_action"]
      ).reshape(-1).astype(np.float32)
      if raw_action_server.shape != (_NUM_JOINTS,) or not np.all(
        np.isfinite(raw_action_server)
      ):
        raise RuntimeError("Invalid raw policy action")
      command_server = _decode_array(
        reply["action"]["joint_position_commands"]["joint_pos"]
      ).reshape(-1).astype(np.float32)
      if command_server.shape != (_NUM_JOINTS,) or not np.all(
        np.isfinite(command_server)
      ):
        raise RuntimeError("Invalid policy command")
      command_hw = np.clip(_server_to_hw(command_server), lower, upper)
      command_step = np.abs(command_hw - prev_cmd_hw)
      max_command_step = float(np.max(command_step))
      if max_command_step > args.max_command_step_rad:
        joint = int(np.argmax(command_step))
        if not args.shadow:
          raise RuntimeError(
            "Command-step guard tripped: "
            f"joint={joint}, delta={max_command_step:.4f} rad"
          )
        elif step == 0:
          print(
            f"[direct] shadow note: step 0 delta from resting pose "
            f"joint={joint}, delta={max_command_step:.4f} rad"
          )

      if not args.shadow:
        bus.write_position(command_hw)
      records.append(
        {
          "elapsed_s": time.monotonic() - started_monotonic,
          "loop_compute_s": time.monotonic() - loop_start,
          "q_hw": q_hw.copy(),
          "qd_hw": qd_hw.copy(),
          "current_ma": current_ma.copy(),
          "temperature_c": temperature_c.copy(),
          "command_hw": command_hw.copy(),
          "raw_action_server": raw_action_server.copy(),
          "tracking_error_rad": tracking_error,
          "stall_recovery_count": stall_recovery_count,
          "stall_recovery_joint": stall_recovery_joint,
          "camera_cube_pose": (
            np.full(9, np.nan, dtype=np.float32)
            if pose_terms is None
            else pose_terms.camera_cube_pose.copy()
          ),
          "pose_goal": (
            np.full(6, np.nan, dtype=np.float32)
            if pose_terms is None
            else pose_terms.pose_goal.copy()
          ),
          "translation_goal_error": (
            np.full(3, np.nan, dtype=np.float32)
            if pose_terms is None
            or pose_terms.translation_goal_error is None
            else pose_terms.translation_goal_error.copy()
          ),
        }
      )
      stall_recovery_joint = -1
      prev_cmd_hw = command_hw
      last_action_server = raw_action_server.copy()
      step += 1

      if step % args.status_interval_steps == 0 or (
        args.max_steps > 0 and step == args.max_steps
      ):
        denominator = str(args.max_steps) if args.max_steps else "∞"
        last_temp = float(np.max(last_temperature_c))
        print(
          f"[direct] step={step}/{denominator} "
          f"max_current={np.max(np.abs(current_ma)):.0f} mA "
          f"max_tracking_error={tracking_error:.3f} rad "
          f"recoveries={stall_recovery_count} "
          f"temperature={last_temp:.0f} C"
        )
      if args.log_chunk_steps and len(records) >= args.log_chunk_steps:
        chunk_path = log_file.with_name(
          f"{log_file.stem}_part{chunk_index:04d}{log_file.suffix}"
        )
        _save_records(
          path=chunk_path,
          records=records,
          started_at=started_at,
          completed=False,
          stop_reason="running",
        )
        saved_log_files.append(chunk_path)
        records.clear()
        chunk_index += 1
      remaining = period - (time.monotonic() - loop_start)
      if remaining > 0:
        time.sleep(remaining)

    completed = True
    stopped_cleanly = True
    stop_reason = "max_steps"
    if args.shadow:
      print("[direct] shadow run completed; no motor command was sent")
    else:
      print(
        f"[direct] completed; returning home over "
        f"{args.return_home_duration_s:.1f} s"
      )
      q_hw, _, _ = bus.read_state()
      _ramp(
        bus,
        q_hw,
        np.zeros(_NUM_JOINTS, dtype=np.float32),
        duration_s=args.return_home_duration_s,
        hz=50.0,
        current_abort_ma=args.current_abort_ma,
        velocity_abort_rad_s=args.velocity_abort_rad_s,
      )
  except KeyboardInterrupt:
    stop_reason = "interrupted"
    if bus.torque_enabled and args.return_home_on_interrupt:
      print(
        f"[direct] interrupted; returning home over "
        f"{args.return_home_duration_s:.1f} s"
      )
      try:
        q_hw, _, _ = bus.read_state()
        _ramp(
          bus,
          q_hw,
          np.zeros(_NUM_JOINTS, dtype=np.float32),
          duration_s=args.return_home_duration_s,
          hz=50.0,
          current_abort_ma=args.current_abort_ma,
          velocity_abort_rad_s=args.velocity_abort_rad_s,
        )
        stopped_cleanly = True
      except Exception as exc:
        print(f"[direct] return-home failed: {exc}; disabling torque")
    else:
      print("[direct] interrupted; disabling torque immediately")
      stopped_cleanly = True
  finally:
    try:
      if bus.torque_enabled:
        bus.disable_torque()
        print("[direct] torque disabled")
    finally:
      bus.close()
      rpc.close()

    if records:
      final_log_file = log_file
      if saved_log_files:
        final_log_file = log_file.with_name(
          f"{log_file.stem}_part{chunk_index:04d}{log_file.suffix}"
        )
      _save_records(
        path=final_log_file,
        records=records,
        started_at=started_at,
        completed=completed,
        stop_reason=stop_reason,
      )
      saved_log_files.append(final_log_file)

  if not completed and not stopped_cleanly:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
