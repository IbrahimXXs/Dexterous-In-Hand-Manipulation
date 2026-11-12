#!/usr/bin/env python3
"""Serve the estimator-augmented yaw policy over ZeroMQ.

Same wire protocol as ``scripts/serve_policy.py`` (ping / init / reset /
infer / close), so ``hardware/deploy_direct.py`` connects unchanged.  Each
``infer`` reply additionally carries ``estimated_cube_state`` (palm frame:
pos, quat_wxyz, lin_vel, ang_vel, contact_prob), which existing clients
ignore and which can be logged for later comparison against an external
camera or motion-capture ground truth (see docs/state_estimation.md).

This server never touches hardware.  Physical runs follow the supervised
procedure (TA_GUIDE.md): start this server, run ``deploy_direct.py --shadow``
first (torque off), and only then a short powered run with the unchanged
current / temperature limits.

    python scripts/serve_estimator_policy.py --bundle checkpoint/estimator/estimator_policy_full.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import zmq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from policy.estimator_policy import EstimatorYawPolicy  # noqa: E402
from scripts.serve_policy import _decode_array, _encode_array  # noqa: E402


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--bundle", type=Path, required=True)
  p.add_argument("--task", default="Mjlab-Leap-Right-HandCube-MBZUAI-Rotate")
  p.add_argument("--bind", default="tcp://127.0.0.1:5555")
  p.add_argument("--keep-alive", action="store_true")
  p.add_argument("--state-log", type=Path, default=None, help="JSONL file for estimated states")
  args = p.parse_args()

  policy = EstimatorYawPolicy(args.bundle)
  print(f"[server] bundle {args.bundle} feature_mode={policy.meta.get('feature_mode')}")
  ctx = zmq.Context.instance()
  sock = ctx.socket(zmq.REP)
  sock.setsockopt(zmq.LINGER, 0)
  sock.bind(args.bind)
  log = open(args.state_log, "a") if args.state_log else None
  print(f"[server] listening on {args.bind}")
  try:
    while True:
      msg = sock.recv_json()
      kind = msg.get("type", "")
      if kind == "ping":
        sock.send_json({"ok": True, "task": args.task})
      elif kind == "init":
        sock.send_json({
          "ok": True,
          "actor_observation_terms": ["joint_pos", "prev_commanded_joint_pos", "joint_pos_error", "last_action"],
          "action_term_metadata": {"joint_pos": {"default_joint_pos": _encode_array(policy.default_joint_pos.astype(np.float32))}},
        })
      elif kind == "reset":
        tv = msg.get("observation", {}).get("term_values", {})
        policy.reset(_decode_array(tv["joint_pos"]).reshape(-1) if "joint_pos" in tv else None)
        sock.send_json({"ok": True})
      elif kind == "infer":
        tv = msg.get("observation", {}).get("term_values", {})
        if "joint_pos" not in tv:
          sock.send_json({"ok": False, "error": "Missing joint_pos in observation"})
          continue
        target, raw, _ = policy.step(_decode_array(tv["joint_pos"]).reshape(-1))
        est = {k: v.astype(np.float32).tolist() for k, v in policy.last_estimate.items()}
        sock.send_json({
          "ok": True,
          "action": {
            "raw_action": _encode_array(np.asarray(raw[None], np.float32)),
            "joint_position_commands": {"joint_pos": _encode_array(np.asarray(target[None], np.float32))},
          },
          "estimated_cube_state": est,
        })
        if log:
          log.write(json.dumps({"t": time.time(), **est}) + "\n")
      elif kind == "close":
        sock.send_json({"ok": True})
        if not args.keep_alive:
          break
      else:
        sock.send_json({"ok": False, "error": f"Unknown message type: {kind}"})
  finally:
    sock.close(linger=0)
    if log:
      log.close()


if __name__ == "__main__":
  main()
