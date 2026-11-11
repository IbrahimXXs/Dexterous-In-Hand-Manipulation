#!/usr/bin/env python3
"""Serve the continuous in-hand yaw rotation policy over ZeroMQ.

Wire-compatible with both CONTINUOUS_Z_ROTATION_DEPLOYMENT_BUNDLE_20261014
and direct client scripts.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any
import numpy as np
import zmq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy.yaw_policy import ContinuousYawPolicy, _NUM_JOINTS


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-file",
        type=Path,
        default=ROOT / "checkpoint" / "model_400.pt",
        help="Path to .pt checkpoint or .npz weights file",
    )
    parser.add_argument(
        "--task",
        default="Mjlab-Leap-Right-HandCube-MBZUAI-Rotate",
        help="Task ID for compatibility check",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device (cpu or cuda)",
    )
    parser.add_argument(
        "--bind",
        default="tcp://127.0.0.1:5555",
        help="ZeroMQ bind endpoint",
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep listening after a client disconnects",
    )
    parser.add_argument(
        "--infer-log-interval",
        type=int,
        default=50,
        help="Steps between status log prints",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[server] Loading policy for task: {args.task}")
    policy = ContinuousYawPolicy(args.checkpoint_file)

    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(args.bind)
    print(f"[server] Bound and listening on {args.bind}")

    step_counter = 0

    try:
        while True:
            msg = socket.recv_json()
            msg_type = msg.get("type", "")

            if msg_type == "ping":
                socket.send_json({"ok": True, "task": args.task})
            elif msg_type == "init":
                # The direct deployment client uses this contract to validate
                # its observations and to ramp safely to the policy's nominal
                # grasp pose before an actuated trial. Keep it explicit so a
                # protocol mismatch fails before the serial bus is opened.
                socket.send_json(
                    {
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
                                    policy.default_joint_pos.astype(np.float32)
                                ),
                            },
                        },
                    }
                )
            elif msg_type == "reset":
                # Reset policy observation buffer with initial state
                obs = msg.get("observation", {})
                term_values = obs.get("term_values", {})
                if "joint_pos" in term_values:
                    q_server = _decode_array(term_values["joint_pos"]).reshape(-1)
                    policy.reset(q_server)
                else:
                    policy.reset()
                step_counter = 0
                socket.send_json({"ok": True})
            elif msg_type == "infer":
                obs = msg.get("observation", {})
                term_values = obs.get("term_values", {})
                if "joint_pos" not in term_values:
                    socket.send_json({"ok": False, "error": "Missing joint_pos in observation"})
                    continue

                q_server = _decode_array(term_values["joint_pos"]).reshape(-1)
                new_target, raw_action, delta = policy.step(q_server)

                row_target = np.asarray(new_target[None, :], dtype=np.float32)
                row_raw = np.asarray(raw_action[None, :], dtype=np.float32)

                reply = {
                    "ok": True,
                    "action": {
                        "raw_action": _encode_array(row_raw),
                        "joint_position_commands": {
                            "joint_pos": _encode_array(row_target),
                        },
                    },
                }
                socket.send_json(reply)

                step_counter += 1
                if step_counter % args.infer_log_interval == 0:
                    print(f"[server] step {step_counter}: max delta={np.max(np.abs(delta)):.4f} rad")
            elif msg_type == "close":
                socket.send_json({"ok": True})
                if not args.keep_alive:
                    print("[server] Client closed connection; shutting down.")
                    break
            else:
                socket.send_json({"ok": False, "error": f"Unknown message type: {msg_type}"})
    finally:
        socket.close(linger=0)
        print("[server] Stopped.")


if __name__ == "__main__":
    main()
