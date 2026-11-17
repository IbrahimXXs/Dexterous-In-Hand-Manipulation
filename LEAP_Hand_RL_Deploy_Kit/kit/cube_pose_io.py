"""Validated file bridge from a camera/goal process to policy deployment.

The producer should atomically replace a JSON file with:

{
  "timestamp_s": 1785763200.0,
  "cube_position_palm_m": [x, y, z],
  "cube_quaternion_palm_wxyz": [w, x, y, z],
  "goal_quaternion_palm_wxyz": [w, x, y, z],
  "goal_position_palm_m": [x, y, z],
  "pickup_phase": "manipulate",
  "pickup_progress": 1.0
}

Both quaternions rotate vectors from the cube/goal frame into the LEAP palm
frame. The camera process is responsible for applying its calibrated
camera-to-palm extrinsic before publishing. goal_position_palm_m is optional
and is used by Task 1 cylinder translation. pickup_phase is optional
for policies that do not use Task 1 pickup. Task 1 requires exactly one of
open, pickup, or manipulate from the interlocked perception supervisor.
pickup_progress is optional for older policies; progress-aware Task 1 policies
require a finite scalar in [0, 1] (open=0, manipulate=1).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CubePosePolicyTerms:
  camera_cube_pose: np.ndarray
  pose_goal: np.ndarray
  camera_cube_position_error: np.ndarray
  camera_cube_orientation_error: np.ndarray
  timestamp_s: float
  translation_goal_error: np.ndarray | None = None
  pickup_phase: np.ndarray | None = None
  pickup_progress: np.ndarray | None = None


def _finite_vector(
  payload: dict[str, Any],
  name: str,
  size: int,
) -> np.ndarray:
  if name not in payload:
    raise ValueError(f"Pose packet is missing '{name}'.")
  value = np.asarray(payload[name], dtype=np.float64)
  if value.shape != (size,):
    raise ValueError(
      f"Pose packet '{name}' must have shape {(size,)}, got {value.shape}."
    )
  if not np.all(np.isfinite(value)):
    raise ValueError(f"Pose packet '{name}' contains non-finite values.")
  return value


def normalize_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
  quaternion = np.asarray(quaternion, dtype=np.float64)
  if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
    raise ValueError("Quaternion must contain four finite wxyz values.")
  norm = float(np.linalg.norm(quaternion))
  if norm < 1e-8:
    raise ValueError("Quaternion norm must be non-zero.")
  return quaternion / norm


def rotation_matrix_from_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
  """Return the active 3x3 rotation matrix for a wxyz quaternion."""
  w, x, y, z = normalize_quaternion_wxyz(quaternion)
  return np.array(
    [
      [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
      [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
      [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ],
    dtype=np.float64,
  )


def rotation_6d_from_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
  """Match MJLab's [r00,r10,r20,r01,r11,r21] rotation-6D convention."""
  rotation = rotation_matrix_from_quaternion_wxyz(quaternion)
  return np.concatenate([rotation[:, 0], rotation[:, 1]]).astype(np.float32)


def decode_pose_packet(
  payload: dict[str, Any],
  *,
  fallback_timestamp_s: float | None = None,
) -> CubePosePolicyTerms:
  position = _finite_vector(payload, "cube_position_palm_m", 3)
  if float(np.linalg.norm(position)) > 1.0:
    raise ValueError(
      "cube_position_palm_m is more than 1 m from the palm; check units/extrinsics."
    )
  cube_quat = _finite_vector(payload, "cube_quaternion_palm_wxyz", 4)
  goal_quat = _finite_vector(payload, "goal_quaternion_palm_wxyz", 4)
  timestamp_s = float(
    payload.get(
      "timestamp_s",
      time.time() if fallback_timestamp_s is None else fallback_timestamp_s,
    )
  )
  if not np.isfinite(timestamp_s):
    raise ValueError("Pose packet timestamp_s must be finite.")

  cube_rotation = rotation_matrix_from_quaternion_wxyz(cube_quat)
  goal_rotation = rotation_matrix_from_quaternion_wxyz(goal_quat)
  camera_cube_pose = np.concatenate(
    [position.astype(np.float32), rotation_6d_from_quaternion_wxyz(cube_quat)]
  ).astype(np.float32)
  pose_goal = rotation_6d_from_quaternion_wxyz(goal_quat)
  relative_rotation = cube_rotation @ goal_rotation.T
  camera_cube_position_error = (-position).astype(np.float32)
  camera_cube_orientation_error = relative_rotation.reshape(9)[3:].astype(
    np.float32
  )
  translation_goal_error = None
  if "goal_position_palm_m" in payload:
    goal_position = _finite_vector(payload, "goal_position_palm_m", 3)
    if float(np.linalg.norm(goal_position)) > 1.0:
      raise ValueError(
        "goal_position_palm_m is more than 1 m from the palm; check units."
      )
    translation_goal_error = (goal_position - position).astype(np.float32)

  pickup_phase = None
  if "pickup_phase" in payload:
    from in_hand_rotation_mjlab.tasks.hand_cube.mdp.cylinder_pickup import (
      pickup_phase_one_hot,
    )

    pickup_phase = (
      pickup_phase_one_hot(payload["pickup_phase"]).cpu().numpy()
    )
  pickup_progress = None
  if "pickup_progress" in payload:
    value = np.asarray(payload["pickup_progress"], dtype=np.float64)
    if value.shape not in ((), (1,)):
      raise ValueError("pickup_progress must be a scalar in [0, 1].")
    scalar = float(value.reshape(-1)[0] if value.shape else value)
    if not np.isfinite(scalar) or not 0.0 <= scalar <= 1.0:
      raise ValueError("pickup_progress must be finite and in [0, 1].")
    pickup_progress = np.asarray([scalar], dtype=np.float32)
  return CubePosePolicyTerms(
    camera_cube_pose=camera_cube_pose,
    pose_goal=pose_goal,
    camera_cube_position_error=camera_cube_position_error,
    camera_cube_orientation_error=camera_cube_orientation_error,
    timestamp_s=timestamp_s,
    translation_goal_error=translation_goal_error,
    pickup_phase=pickup_phase,
    pickup_progress=pickup_progress,
  )


class JsonCubePoseSource:
  """Read and freshness-check an atomically updated JSON pose packet."""

  def __init__(self, path: Path, max_age_s: float):
    self.path = Path(path)
    self.max_age_s = float(max_age_s)
    if self.max_age_s <= 0.0:
      raise ValueError("max_age_s must be positive.")

  def read(self, *, now_s: float | None = None) -> CubePosePolicyTerms:
    if not self.path.is_file():
      raise FileNotFoundError(f"Cube pose packet not found: {self.path}")
    stat = self.path.stat()
    with self.path.open("r", encoding="utf-8") as stream:
      payload = json.load(stream)
    if not isinstance(payload, dict):
      raise ValueError("Cube pose packet root must be a JSON object.")
    terms = decode_pose_packet(payload, fallback_timestamp_s=stat.st_mtime)
    now = time.time() if now_s is None else float(now_s)
    age_s = now - terms.timestamp_s
    if age_s < -1.0:
      raise ValueError(
        f"Cube pose timestamp is {-age_s:.3f} s in the future; clocks disagree."
      )
    if age_s > self.max_age_s:
      raise RuntimeError(
        f"Cube pose packet is stale by {age_s:.3f} s "
        f"(limit {self.max_age_s:.3f} s)."
      )
    return terms
