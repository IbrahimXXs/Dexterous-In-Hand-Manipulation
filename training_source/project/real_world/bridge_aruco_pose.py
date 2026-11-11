#!/usr/bin/env python3
"""Bridge RealSense ArUco JSONL poses into the LEAP policy pose packet.

The fixed-camera calibration is ``T_palm_camera``:

  p_palm = R_palm_camera @ p_camera + t_palm_camera
  R_palm_cube = R_palm_camera @ R_camera_cube

The output is atomically replaced so the 20 Hz hardware client never observes
a partially written JSON document.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

try:
  from .cube_pose_io import (
    normalize_quaternion_wxyz,
    rotation_matrix_from_quaternion_wxyz,
  )
except ImportError:
  from cube_pose_io import (
    normalize_quaternion_wxyz,
    rotation_matrix_from_quaternion_wxyz,
  )


def _vector(payload: dict[str, Any], name: str, size: int) -> np.ndarray:
  value = np.asarray(payload.get(name), dtype=np.float64)
  if value.shape != (size,) or not np.all(np.isfinite(value)):
    raise ValueError(f"'{name}' must contain {size} finite values")
  return value


def _rotation_matrix(value: Any, name: str) -> np.ndarray:
  matrix = np.asarray(value, dtype=np.float64)
  if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
    raise ValueError(f"'{name}' must be a finite 3x3 matrix")
  if not np.allclose(matrix.T @ matrix, np.eye(3), atol=2e-3):
    raise ValueError(f"'{name}' is not orthonormal")
  if not np.isclose(np.linalg.det(matrix), 1.0, atol=2e-3):
    raise ValueError(f"'{name}' must have determinant +1")
  return matrix


def quaternion_wxyz_from_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
  """Convert a proper 3x3 rotation matrix to a normalized wxyz quaternion."""
  matrix = _rotation_matrix(matrix, "rotation_matrix")
  trace = float(np.trace(matrix))
  if trace > 0.0:
    scale = 2.0 * np.sqrt(trace + 1.0)
    quaternion = np.array(
      [
        0.25 * scale,
        (matrix[2, 1] - matrix[1, 2]) / scale,
        (matrix[0, 2] - matrix[2, 0]) / scale,
        (matrix[1, 0] - matrix[0, 1]) / scale,
      ]
    )
  else:
    index = int(np.argmax(np.diag(matrix)))
    if index == 0:
      scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
      quaternion = np.array(
        [
          (matrix[2, 1] - matrix[1, 2]) / scale,
          0.25 * scale,
          (matrix[0, 1] + matrix[1, 0]) / scale,
          (matrix[0, 2] + matrix[2, 0]) / scale,
        ]
      )
    elif index == 1:
      scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
      quaternion = np.array(
        [
          (matrix[0, 2] - matrix[2, 0]) / scale,
          (matrix[0, 1] + matrix[1, 0]) / scale,
          0.25 * scale,
          (matrix[1, 2] + matrix[2, 1]) / scale,
        ]
      )
    else:
      scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
      quaternion = np.array(
        [
          (matrix[1, 0] - matrix[0, 1]) / scale,
          (matrix[0, 2] + matrix[2, 0]) / scale,
          (matrix[1, 2] + matrix[2, 1]) / scale,
          0.25 * scale,
        ]
      )
  quaternion = normalize_quaternion_wxyz(quaternion)
  return -quaternion if quaternion[0] < 0.0 else quaternion


def load_camera_to_palm(path: Path) -> tuple[np.ndarray, np.ndarray]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise ValueError("Camera-to-palm calibration must be a JSON object")
  if payload.get("calibrated") is not True:
    raise ValueError(
      "Camera-to-palm file is not marked calibrated=true; do not deploy an "
      "example or guessed transform"
    )
  translation = _vector(payload, "translation_palm_camera_m", 3)
  quaternion = _vector(payload, "quaternion_palm_camera_wxyz", 4)
  rotation = rotation_matrix_from_quaternion_wxyz(quaternion)
  return translation, rotation


def latest_jsonl_record(path: Path, read_bytes: int = 262_144) -> dict[str, Any]:
  """Read the newest complete JSON object from an actively appended JSONL."""
  with path.open("rb") as stream:
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    stream.seek(max(0, size - read_bytes), os.SEEK_SET)
    lines = stream.read().splitlines()
  for line in reversed(lines):
    try:
      payload = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError):
      continue
    if isinstance(payload, dict):
      return payload
  raise ValueError(f"No complete JSON record found in {path}")


def goal_quaternion(
  goal_file: Path | None,
  fallback_wxyz: tuple[float, float, float, float],
) -> np.ndarray:
  if goal_file is None:
    return normalize_quaternion_wxyz(np.asarray(fallback_wxyz))
  payload = json.loads(goal_file.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise ValueError("Goal file must be a JSON object")
  return normalize_quaternion_wxyz(
    _vector(payload, "goal_quaternion_palm_wxyz", 4)
  )


def convert_tracker_record(
  record: dict[str, Any],
  translation_palm_camera: np.ndarray,
  rotation_palm_camera: np.ndarray,
  goal_quaternion_palm_wxyz: np.ndarray,
  goal_position_palm_m: np.ndarray | None = None,
) -> dict[str, Any]:
  if record.get("tracking_valid", True) is not True:
    raise ValueError("Latest tracker record is not valid")
  position_camera = _vector(record, "cube_center_m", 3)
  if "cube_rotation_matrix" in record:
    rotation_camera_cube = _rotation_matrix(
      record["cube_rotation_matrix"], "cube_rotation_matrix"
    )
  else:
    xyzw = _vector(record, "cube_quaternion_xyzw", 4)
    rotation_camera_cube = rotation_matrix_from_quaternion_wxyz(
      np.roll(xyzw, 1)
    )

  position_palm = (
    rotation_palm_camera @ position_camera + translation_palm_camera
  )
  rotation_palm_cube = rotation_palm_camera @ rotation_camera_cube
  cube_quaternion_palm = quaternion_wxyz_from_rotation_matrix(
    rotation_palm_cube
  )
  timestamp_s = float(record.get("host_time_s", time.time()))
  if not np.isfinite(timestamp_s):
    raise ValueError("Tracker host_time_s is not finite")
  packet = {
    "timestamp_s": timestamp_s,
    "cube_position_palm_m": position_palm.tolist(),
    "cube_quaternion_palm_wxyz": cube_quaternion_palm.tolist(),
    "goal_quaternion_palm_wxyz": normalize_quaternion_wxyz(
      goal_quaternion_palm_wxyz
    ).tolist(),
    "source_frame": int(record.get("frame", -1)),
    "source_visible_marker_ids": record.get("visible_marker_ids", []),
    "source_reprojection_error_px": record.get("reprojection_error_px"),
  }
  if goal_position_palm_m is not None:
    goal_position = np.asarray(goal_position_palm_m, dtype=np.float64)
    if goal_position.shape != (3,) or not np.all(np.isfinite(goal_position)):
      raise ValueError("goal_position_palm_m must contain three finite values")
    packet["goal_position_palm_m"] = goal_position.tolist()
  return packet


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, temporary_name = tempfile.mkstemp(
    prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
  )
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
      json.dump(payload, stream, separators=(",", ":"))
      stream.write("\n")
      stream.flush()
      os.fsync(stream.fileno())
    os.replace(temporary_name, path)
  finally:
    if os.path.exists(temporary_name):
      os.unlink(temporary_name)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--tracker-jsonl", type=Path, required=True)
  parser.add_argument("--camera-to-palm", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--goal-file", type=Path)
  parser.add_argument(
    "--goal-quaternion-palm-wxyz",
    type=float,
    nargs=4,
    default=(1.0, 0.0, 0.0, 0.0),
  )
  parser.add_argument(
    "--goal-position-palm-m",
    type=float,
    nargs=3,
    default=None,
    help="Optional absolute palm-frame XYZ target for translation policies.",
  )
  parser.add_argument("--max-record-age-s", type=float, default=0.25)
  parser.add_argument("--poll-hz", type=float, default=60.0)
  parser.add_argument("--once", action="store_true")
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  if args.max_record_age_s <= 0.0 or args.poll_hz <= 0.0:
    raise SystemExit("--max-record-age-s and --poll-hz must be positive")
  translation, rotation = load_camera_to_palm(args.camera_to_palm)
  last_frame: int | None = None
  period = 1.0 / args.poll_hz
  while True:
    try:
      record = latest_jsonl_record(args.tracker_jsonl)
      timestamp_s = float(record.get("host_time_s", 0.0))
      age_s = time.time() - timestamp_s
      if age_s > args.max_record_age_s:
        raise RuntimeError(
          f"latest tracker pose is stale by {age_s:.3f} s "
          f"(limit {args.max_record_age_s:.3f} s)"
        )
      frame = int(record.get("frame", -1))
      if frame != last_frame:
        goal = goal_quaternion(
          args.goal_file, tuple(args.goal_quaternion_palm_wxyz)
        )
        goal_position = (
          None
          if args.goal_position_palm_m is None
          else np.asarray(args.goal_position_palm_m, dtype=np.float64)
        )
        if args.goal_file is not None:
          goal_payload = json.loads(
            args.goal_file.read_text(encoding="utf-8")
          )
          if "goal_position_palm_m" in goal_payload:
            goal_position = _vector(
              goal_payload, "goal_position_palm_m", 3
            )
        packet = convert_tracker_record(
          record,
          translation,
          rotation,
          goal,
          goal_position_palm_m=goal_position,
        )
        atomic_write_json(args.output, packet)
        last_frame = frame
        if args.once:
          print(json.dumps(packet, indent=2))
          return 0
    except (FileNotFoundError, ValueError, RuntimeError) as error:
      if args.once:
        raise
      print(f"[cube-pose-bridge] waiting: {error}", flush=True)
    time.sleep(period)


if __name__ == "__main__":
  raise SystemExit(main())
