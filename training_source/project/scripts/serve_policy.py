#!/usr/bin/env python3
"""Serve an MJLab checkpoint to the guarded real-world ZMQ client."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from in_hand_rotation_mjlab.policy_server.config import PolicyServerConfig
from in_hand_rotation_mjlab.policy_server.zmq_transport import (
  PolicyZmqServer,
  ZmqServerConfig,
)
from mjlab.utils.torch import configure_torch_backends

import mjlab.tasks  # noqa: F401
import in_hand_rotation_mjlab.tasks  # noqa: F401


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint-file", type=Path, required=True)
  parser.add_argument(
    "--task",
    default="Mjlab-Leap-Right-HandCube-MBZUAI-Visual-Yaw",
  )
  parser.add_argument(
    "--device",
    default=None,
    help="Inference device; default is cuda:0 when available, otherwise cpu.",
  )
  parser.add_argument("--bind", default="tcp://127.0.0.1:5555")
  parser.add_argument("--infer-log-interval", type=int, default=50)
  parser.add_argument(
    "--keep-alive",
    action="store_true",
    help="Keep listening after a client sends close.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  checkpoint = args.checkpoint_file.expanduser().resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
  if args.infer_log_interval <= 0:
    raise ValueError("--infer-log-interval must be positive")

  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  server = PolicyZmqServer(
    task_id=args.task,
    cfg=PolicyServerConfig(
      checkpoint_file=str(checkpoint),
      device=device,
      play=True,
    ),
    transport_cfg=ZmqServerConfig(
      bind=args.bind,
      shutdown_on_close=not args.keep_alive,
      infer_log_interval=args.infer_log_interval,
    ),
  )
  server.serve_forever()


if __name__ == "__main__":
  main()
