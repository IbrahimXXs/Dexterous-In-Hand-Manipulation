"""Validated, dropout-aware filtering for the live tracked-cube pose stream."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

try:
  from .cube_pose_io import decode_pose_packet, normalize_quaternion_wxyz
except ImportError:
  from cube_pose_io import decode_pose_packet, normalize_quaternion_wxyz


@dataclass(frozen=True)
class FilteredCubePose:
  position_palm_m: np.ndarray
  quaternion_palm_wxyz: np.ndarray
  timestamp_s: float
  source_frame: int | None
  visible_marker_ids: tuple[int, ...]
  goal_quaternion_palm_wxyz: np.ndarray | None = None


@dataclass(frozen=True)
class PosePollResult:
  pose: FilteredCubePose | None
  state: str
  message: str
  age_s: float | None


def quaternion_angle_rad(first: np.ndarray, second: np.ndarray) -> float:
  """Return the shortest SO(3) angle between two WXYZ quaternions."""
  first = normalize_quaternion_wxyz(first)
  second = normalize_quaternion_wxyz(second)
  dot = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
  return 2.0 * math.acos(dot)


def quaternion_slerp(
  first: np.ndarray,
  second: np.ndarray,
  fraction: float,
) -> np.ndarray:
  """Shortest-path normalized quaternion interpolation."""
  first = normalize_quaternion_wxyz(first)
  second = normalize_quaternion_wxyz(second)
  dot = float(np.dot(first, second))
  if dot < 0.0:
    second = -second
    dot = -dot
  fraction = float(np.clip(fraction, 0.0, 1.0))
  if dot > 0.9995:
    return normalize_quaternion_wxyz(first + fraction * (second - first))
  angle = math.acos(float(np.clip(dot, -1.0, 1.0)))
  scale = math.sin(angle)
  return normalize_quaternion_wxyz(
    math.sin((1.0 - fraction) * angle) / scale * first
    + math.sin(fraction * angle) / scale * second
  )


class RobustCubePoseFilter:
  """Low-pass pose data while rejecting physically implausible jumps."""

  def __init__(
    self,
    *,
    position_time_constant_s: float = 0.035,
    orientation_time_constant_s: float = 0.025,
    max_linear_speed_m_s: float = 2.5,
    max_angular_speed_rad_s: float = 18.0,
  ) -> None:
    if position_time_constant_s < 0.0 or orientation_time_constant_s < 0.0:
      raise ValueError("Filter time constants must be non-negative.")
    if max_linear_speed_m_s <= 0.0 or max_angular_speed_rad_s <= 0.0:
      raise ValueError("Pose speed limits must be positive.")
    self.position_time_constant_s = float(position_time_constant_s)
    self.orientation_time_constant_s = float(orientation_time_constant_s)
    self.max_linear_speed_m_s = float(max_linear_speed_m_s)
    self.max_angular_speed_rad_s = float(max_angular_speed_rad_s)
    self._pose: FilteredCubePose | None = None

  @property
  def pose(self) -> FilteredCubePose | None:
    return self._pose

  @staticmethod
  def _alpha(dt: float, time_constant: float) -> float:
    if time_constant <= 0.0:
      return 1.0
    return 1.0 - math.exp(-dt / time_constant)

  def update(
    self,
    *,
    position_palm_m: np.ndarray,
    quaternion_palm_wxyz: np.ndarray,
    timestamp_s: float,
    source_frame: int | None = None,
    visible_marker_ids: tuple[int, ...] = (),
  ) -> FilteredCubePose:
    position = np.asarray(position_palm_m, dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
      raise ValueError("Tracked position must contain three finite values.")
    quaternion = normalize_quaternion_wxyz(quaternion_palm_wxyz)
    timestamp_s = float(timestamp_s)
    if not np.isfinite(timestamp_s):
      raise ValueError("Tracked pose timestamp must be finite.")

    if self._pose is None:
      self._pose = FilteredCubePose(
        position,
        quaternion,
        timestamp_s,
        source_frame,
        visible_marker_ids,
      )
      return self._pose

    dt = timestamp_s - self._pose.timestamp_s
    if dt < -1e-6:
      raise ValueError("Tracked pose timestamps moved backwards.")
    if dt <= 1e-6:
      return self._pose

    linear_speed = float(np.linalg.norm(position - self._pose.position_palm_m)) / dt
    angular_speed = quaternion_angle_rad(
      quaternion,
      self._pose.quaternion_palm_wxyz,
    ) / dt
    if linear_speed > self.max_linear_speed_m_s:
      raise ValueError(
        f"Rejected pose jump: {linear_speed:.2f} m/s exceeds "
        f"{self.max_linear_speed_m_s:.2f} m/s."
      )
    if angular_speed > self.max_angular_speed_rad_s:
      raise ValueError(
        f"Rejected pose jump: {math.degrees(angular_speed):.0f} deg/s exceeds "
        f"{math.degrees(self.max_angular_speed_rad_s):.0f} deg/s."
      )

    position_alpha = self._alpha(dt, self.position_time_constant_s)
    orientation_alpha = self._alpha(dt, self.orientation_time_constant_s)
    self._pose = FilteredCubePose(
      self._pose.position_palm_m
      + position_alpha * (position - self._pose.position_palm_m),
      quaternion_slerp(
        self._pose.quaternion_palm_wxyz,
        quaternion,
        orientation_alpha,
      ),
      timestamp_s,
      source_frame,
      visible_marker_ids,
    )
    return self._pose


class FilteredJsonCubePoseSource:
  """Poll an atomic JSON packet and return TRACKING, HOLD, or LOST."""

  def __init__(
    self,
    path: Path,
    *,
    max_age_s: float = 0.25,
    hold_s: float = 0.50,
    pose_filter: RobustCubePoseFilter | None = None,
  ) -> None:
    self.path = Path(path)
    self.max_age_s = float(max_age_s)
    self.hold_s = float(hold_s)
    if self.max_age_s <= 0.0 or self.hold_s < self.max_age_s:
      raise ValueError("hold_s must be at least max_age_s, and max_age_s > 0.")
    self.pose_filter = pose_filter or RobustCubePoseFilter()
    self._last_packet_timestamp_s: float | None = None
    self._last_error = "waiting for first pose packet"
    self._last_goal_quaternion: np.ndarray | None = None

  @staticmethod
  def _optional_ids(payload: dict[str, Any]) -> tuple[int, ...]:
    values = payload.get("source_visible_marker_ids", ())
    if not isinstance(values, (list, tuple)):
      return ()
    try:
      return tuple(int(value) for value in values)
    except (TypeError, ValueError):
      return ()

  def poll(self, *, now_s: float | None = None) -> PosePollResult:
    now = time.time() if now_s is None else float(now_s)
    try:
      payload = json.loads(self.path.read_text(encoding="utf-8"))
      if not isinstance(payload, dict):
        raise ValueError("Pose packet root must be a JSON object.")
      if payload.get("tracking_valid", True) is not True:
        raise ValueError("Tracker marked the pose invalid.")
      terms = decode_pose_packet(payload)
      age_s = now - terms.timestamp_s
      if age_s < -1.0:
        raise ValueError(f"Pose timestamp is {-age_s:.3f} s in the future.")
      if age_s > self.max_age_s:
        raise RuntimeError(f"Pose packet is stale by {age_s:.3f} s.")
      quaternion = normalize_quaternion_wxyz(
        np.asarray(payload["cube_quaternion_palm_wxyz"], dtype=np.float64)
      )
      goal_quaternion = normalize_quaternion_wxyz(
        np.asarray(payload["goal_quaternion_palm_wxyz"], dtype=np.float64)
      )
      self._last_goal_quaternion = goal_quaternion
      if terms.timestamp_s != self._last_packet_timestamp_s:
        self.pose_filter.update(
          position_palm_m=terms.camera_cube_pose[:3],
          quaternion_palm_wxyz=quaternion,
          timestamp_s=terms.timestamp_s,
          source_frame=(
            int(payload["source_frame"])
            if payload.get("source_frame") is not None
            else None
          ),
          visible_marker_ids=self._optional_ids(payload),
        )
        self._last_packet_timestamp_s = terms.timestamp_s
      self._last_error = ""
      filtered_pose = self.pose_filter.pose
      if filtered_pose is not None:
        filtered_pose = replace(
          filtered_pose,
          goal_quaternion_palm_wxyz=goal_quaternion.copy(),
        )
      return PosePollResult(
        filtered_pose,
        "TRACKING",
        "fresh filtered pose",
        age_s,
      )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError) as error:
      self._last_error = str(error)

    pose = self.pose_filter.pose
    age_s = None if pose is None else now - pose.timestamp_s
    if pose is not None and self._last_goal_quaternion is not None:
      pose = replace(
        pose,
        goal_quaternion_palm_wxyz=self._last_goal_quaternion.copy(),
      )
    if pose is not None and age_s <= self.hold_s:
      return PosePollResult(pose, "HOLD", self._last_error, age_s)
    return PosePollResult(None, "LOST", self._last_error, age_s)
