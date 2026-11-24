"""Camera intrinsics: documented JSON format, validation, and honest provenance.

File format (``*.intrinsics.json``)::

  {
    "format": "leap_cube_vision_intrinsics/v1",
    "camera_id": "realsense:D435I:123456789",      # free text, as specific as possible
    "status": "checkerboard_calibrated",            # see STATUS_* below
    "image_size_wh": [1280, 720],
    "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "dist_coeffs": [k1, k2, p1, p2, k3],            # OpenCV model
    "rms_reprojection_px": 0.31,                     # calibration result (if calibrated)
    "n_views": 24, "board": {...}, "created": "...", "notes": "...",
    "depth": {                                       # RGB-D cameras only
      "depth_scale_m_per_unit": 0.001,
      "aligned_to_color": true,
      "source": "pyrealsense2 rs.align(rs.stream.color)"
    }
  }

There is NO default camera matrix: a source without a valid intrinsics file cannot produce
metric poses (the tracker refuses), it never falls back to a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import datetime as _dt
import json
from pathlib import Path
from typing import Any

import numpy as np

FORMAT = "leap_cube_vision_intrinsics/v1"
STATUS_CHECKERBOARD = "checkerboard_calibrated"   # this tool's checkerboard / ChArUco calibration
STATUS_DEVICE = "device_factory_unverified"       # e.g. RealSense SDK intrinsics, not independently checked
STATUS_DEVICE_VERIFIED = "device_factory_verified"  # factory values checked against a board calibration
STATUS_MARKER_ESTIMATE = "estimated_from_cube_markers"  # self-calibration from the cube in a recording
STATUS_UNCALIBRATED = "uncalibrated"
METRIC_STATUSES = (STATUS_CHECKERBOARD, STATUS_DEVICE, STATUS_DEVICE_VERIFIED, STATUS_MARKER_ESTIMATE)
VERIFIED_STATUSES = (STATUS_CHECKERBOARD, STATUS_DEVICE_VERIFIED)


class IntrinsicsError(ValueError):
  pass


@dataclass(frozen=True)
class CameraIntrinsics:
  camera_matrix: np.ndarray
  dist_coeffs: np.ndarray
  image_size_wh: tuple[int, int]
  status: str
  camera_id: str = "unknown"
  rms_reprojection_px: float | None = None
  depth_scale_m_per_unit: float | None = None
  depth_aligned_to_color: bool | None = None
  source_path: str | None = None
  extra: dict[str, Any] = field(default_factory=dict)

  @property
  def fx(self) -> float:
    return float(self.camera_matrix[0, 0])

  @property
  def fy(self) -> float:
    return float(self.camera_matrix[1, 1])

  @property
  def cx(self) -> float:
    return float(self.camera_matrix[0, 2])

  @property
  def cy(self) -> float:
    return float(self.camera_matrix[1, 2])

  @property
  def metric(self) -> bool:
    return self.status in METRIC_STATUSES

  @property
  def verified(self) -> bool:
    return self.status in VERIFIED_STATUSES

  def label(self) -> str:
    return {
      STATUS_CHECKERBOARD: "intrinsics: board-calibrated",
      STATUS_DEVICE: "intrinsics: device factory (unverified)",
      STATUS_DEVICE_VERIFIED: "intrinsics: device factory (verified)",
      STATUS_MARKER_ESTIMATE: "intrinsics: ESTIMATED (approx.)",
    }.get(self.status, "intrinsics: UNCALIBRATED")

  def check_frame(self, width: int, height: int) -> None:
    if (int(width), int(height)) != tuple(self.image_size_wh):
      raise IntrinsicsError(
        f"intrinsics are for {self.image_size_wh[0]}x{self.image_size_wh[1]} but the frames are {width}x{height}; "
        "recalibrate at this resolution (intrinsics are never rescaled silently)")

  def to_json(self) -> dict[str, Any]:
    d = {"format": FORMAT, "camera_id": self.camera_id, "status": self.status,
         "image_size_wh": list(self.image_size_wh), "camera_matrix": self.camera_matrix.tolist(),
         "dist_coeffs": self.dist_coeffs.reshape(-1).tolist(), "rms_reprojection_px": self.rms_reprojection_px}
    if self.depth_scale_m_per_unit is not None:
      d["depth"] = {"depth_scale_m_per_unit": self.depth_scale_m_per_unit, "aligned_to_color": self.depth_aligned_to_color}
    d.update(self.extra)
    return d

  def save(self, path: Path) -> None:
    payload = self.to_json()
    payload.setdefault("created", _dt.datetime.now().astimezone().isoformat(timespec="seconds"))
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def validate(K: np.ndarray, dist: np.ndarray, size_wh) -> None:
  K = np.asarray(K, dtype=np.float64)
  if K.shape != (3, 3) or not np.all(np.isfinite(K)):
    raise IntrinsicsError("camera_matrix must be a finite 3x3 matrix")
  if not (K[0, 0] > 0 and K[1, 1] > 0):
    raise IntrinsicsError("focal lengths fx, fy must be positive")
  if not np.allclose(K[2], [0, 0, 1]) or abs(K[1, 0]) > 1e-9 or abs(K[0, 1]) > 1e-6 * K[0, 0]:
    raise IntrinsicsError("camera_matrix must have the form [[fx,0,cx],[0,fy,cy],[0,0,1]]")
  w, h = (int(v) for v in size_wh)
  if w <= 0 or h <= 0:
    raise IntrinsicsError("image_size_wh must be positive")
  if not (0 < K[0, 2] < w and 0 < K[1, 2] < h):
    raise IntrinsicsError("principal point lies outside the image")
  if not (0.2 < K[0, 0] / K[1, 1] < 5):
    raise IntrinsicsError("fx/fy ratio is implausible")
  dist = np.asarray(dist, dtype=np.float64).reshape(-1)
  if dist.size not in (4, 5, 8, 12, 14) or not np.all(np.isfinite(dist)):
    raise IntrinsicsError("dist_coeffs must hold 4, 5, 8, 12 or 14 finite OpenCV coefficients")


def load_intrinsics(path: Path) -> CameraIntrinsics:
  path = Path(path)
  if not path.is_file():
    raise IntrinsicsError(f"intrinsics file not found: {path}")
  d = json.loads(path.read_text(encoding="utf-8"))
  if d.get("format") != FORMAT:
    raise IntrinsicsError(f"{path}: not a {FORMAT} file")
  status = str(d.get("status", STATUS_UNCALIBRATED))
  if "camera_matrix" not in d or d.get("camera_matrix") is None:
    raise IntrinsicsError(f"{path}: no camera_matrix (status {status}); calibrate the camera first "
                          "(scripts/cube_vision.py calibrate-intrinsics)")
  K = np.asarray(d["camera_matrix"], dtype=np.float64)
  dist = np.asarray(d.get("dist_coeffs", []), dtype=np.float64).reshape(-1)
  size = tuple(int(v) for v in d.get("image_size_wh", (0, 0)))
  validate(K, dist, size)
  if status not in METRIC_STATUSES:
    raise IntrinsicsError(f"{path}: status {status!r} is not usable for metric pose")
  depth = d.get("depth") or {}
  known = {"format", "camera_id", "status", "image_size_wh", "camera_matrix", "dist_coeffs", "rms_reprojection_px", "depth"}
  return CameraIntrinsics(K, dist, size, status, str(d.get("camera_id", "unknown")), d.get("rms_reprojection_px"),
                          depth.get("depth_scale_m_per_unit"), depth.get("aligned_to_color"), str(path),
                          {k: v for k, v in d.items() if k not in known})


def with_status(intr: CameraIntrinsics, status: str) -> CameraIntrinsics:
  return replace(intr, status=status)
