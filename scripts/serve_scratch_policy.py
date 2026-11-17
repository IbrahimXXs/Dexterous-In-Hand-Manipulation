#!/usr/bin/env python3
"""Serve a from-scratch study policy (A baseline or C estimator-augmented) over ZeroMQ.

Same wire protocol as ``scripts/serve_policy.py`` (ping / init / reset / infer /
close), so ``hardware/deploy_direct.py`` connects unchanged.  Each ``infer``
reply also carries the monitors (estimated cube state, fingertip forces), which
the bridge ignores.  The server records every step to ``--log-file`` (npz):
the exact 320-D observation, raw and executed action, target, estimated cube
state, the 19 cube features and fingertip forces, plus reset markers.

This server never touches hardware.  Physical runs follow TA_GUIDE.md: start
this server, run ``deploy_direct.py --shadow`` first (torque off), and only
then a short powered run with the unchanged current / temperature limits.

    uv run python scripts/serve_scratch_policy.py --bundle checkpoint/scratch_study/C_seed2.pt \
        --log-file outputs/ab_test/server_C_seed2_trial01.npz
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

from policy.scratch_policy import ScratchStudyPolicy  # noqa: E402
from scripts.serve_policy import _decode_array, _encode_array  # noqa: E402

_FIELDS = ("obs", "action", "cube_state", "features", "forces")


class StepLog:
  def __init__(self, path: Path | None, meta: dict):
    self.path, self.meta = path, meta
    self.rows: list[dict] = []
    self.resets: list[int] = []

  def reset(self) -> None:
    self.resets.append(len(self.rows))

  def add(self, q, raw, target, outputs) -> None:
    self.rows.append({"t": time.time(), "q": q, "raw": raw, "target": target, **{k: outputs[k] for k in _FIELDS}})

  def save(self) -> None:
    if self.path is None or not self.rows or getattr(self, "_saved_n", -1) == len(self.rows):
      return
    self._saved_n = len(self.rows)
    self.path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
      self.path,
      meta=np.array(json.dumps(self.meta)),
      t_wall=np.array([r["t"] for r in self.rows]),
      reset_at_step=np.array(self.resets, dtype=np.int64),
      q_server=np.stack([r["q"] for r in self.rows]),
      raw_action=np.stack([r["raw"] for r in self.rows]),
      target_server=np.stack([r["target"] for r in self.rows]),
      **{k: np.stack([r[k] for r in self.rows]) for k in _FIELDS},
    )
    print(f"[server] saved {len(self.rows)} steps -> {self.path}")


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--bundle", type=Path, required=True, help="checkpoint/scratch_study/{A,C}_seedN.pt")
  p.add_argument("--task", default="Mjlab-Leap-Scratch")
  p.add_argument("--bind", default="tcp://127.0.0.1:5555")
  p.add_argument("--keep-alive", action="store_true")
  p.add_argument("--log-file", type=Path, default=None, help="npz written on every close and on exit")
  p.add_argument("--print-interval", type=int, default=40)
  args = p.parse_args()

  policy = ScratchStudyPolicy(args.bundle)
  meta = {**policy.meta, "bundle_path": str(args.bundle), "server_started": time.strftime("%Y-%m-%dT%H:%M:%S")}
  print(f"[server] {args.bundle.name}: condition {policy.condition} "
        f"({meta['estimator_role']}), squash={meta['action_squash']}, src={meta['source_checkpoint']}")
  log = StepLog(args.log_file, meta)

  ctx = zmq.Context.instance()
  sock = ctx.socket(zmq.REP)
  sock.setsockopt(zmq.LINGER, 0)
  sock.bind(args.bind)
  print(f"[server] listening on {args.bind}")
  n = 0
  try:
    while True:
      msg = sock.recv_json()
      kind = msg.get("type", "")
      if kind == "ping":
        sock.send_json({"ok": True, "task": args.task, "condition": policy.condition})
      elif kind == "init":
        sock.send_json({
          "ok": True,
          "actor_observation_terms": ["joint_pos", "prev_commanded_joint_pos", "joint_pos_error", "last_action"],
          "action_term_metadata": {"joint_pos": {"default_joint_pos": _encode_array(policy.default_joint_pos.astype(np.float32))}},
        })
      elif kind == "reset":
        tv = msg.get("observation", {}).get("term_values", {})
        policy.reset(_decode_array(tv["joint_pos"]).reshape(-1) if "joint_pos" in tv else None)
        log.reset()
        sock.send_json({"ok": True})
      elif kind == "infer":
        tv = msg.get("observation", {}).get("term_values", {})
        if "joint_pos" not in tv:
          sock.send_json({"ok": False, "error": "Missing joint_pos in observation"})
          continue
        q = _decode_array(tv["joint_pos"]).reshape(-1).astype(np.float32)
        target, raw, _ = policy.step(q)
        out = policy.last_outputs
        log.add(q, raw, target, out)
        sock.send_json({
          "ok": True,
          "action": {
            "raw_action": _encode_array(np.asarray(raw[None], np.float32)),
            "joint_position_commands": {"joint_pos": _encode_array(np.asarray(target[None], np.float32))},
          },
          "estimated_cube_state": out["cube_state"].astype(float).tolist(),
          "estimated_fingertip_forces_N": out["forces"].astype(float).tolist(),
        })
        n += 1
        if args.print_interval and n % args.print_interval == 0:
          s, f = out["cube_state"], out["forces"]
          print(f"[server] step {n}: |tanh|>0.99 {np.mean(np.abs(out['action']) > 0.99):.2f}  "
                f"est pos(mm) {np.round(s[:3] * 1000, 1)}  contacts {np.round(s[13:17], 2)}  "
                f"forces(N) th/if/mf/rf {np.round(f, 2)}")
      elif kind == "close":
        sock.send_json({"ok": True})
        log.save()
        if not args.keep_alive:
          break
      else:
        sock.send_json({"ok": False, "error": f"Unknown message type: {kind}"})
  except KeyboardInterrupt:
    pass
  finally:
    log.save()
    sock.close(linger=0)
    print("[server] stopped")


if __name__ == "__main__":
  main()
