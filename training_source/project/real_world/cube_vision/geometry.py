"""MBZUAI cube marker geometry, loaded from the project's layout files (never duplicated).

* ``assets/mbzuai_cube/layout.json``  - authoritative: cube size, marker size, dictionary,
  per-face IDs and the marker x / y axes in the cube frame.
* ``assets/mbzuai_cube/marker_print_layout.json`` - where the marker square sits inside its
  face (layout.json does not say; the printed markers are NOT centred).

Frames
------
cube frame: origin at the cube centre, +X = B, +Y = Z, +Z = M (layout.json ``body_frame``).
marker frame (OpenCV convention): x right, y up, z out of the face; OpenCV returns the four
corners clockwise starting top-left: TL, TR, BR, BL = (-s/2, +s/2), (+s/2, +s/2),
(+s/2, -s/2), (-s/2, -s/2).  The face's outward normal is therefore marker_x x marker_y.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
LAYOUT_PATH = PROJECT / "assets" / "mbzuai_cube" / "layout.json"
PRINT_LAYOUT_PATH = PROJECT / "assets" / "mbzuai_cube" / "marker_print_layout.json"

#: marker-frame corner coordinates in units of the marker side, OpenCV order TL, TR, BR, BL
CORNERS_UNIT = np.array([[-0.5, 0.5], [0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])

_AXIS_NAMES = {"+X": (1, 0, 0), "-X": (-1, 0, 0), "+Y": (0, 1, 0), "-Y": (0, -1, 0),
               "+Z": (0, 0, 1), "-Z": (0, 0, -1)}


class LayoutError(ValueError):
  """The layout files are inconsistent or incomplete."""


@dataclass(frozen=True)
class FaceSpec:
  marker_id: int
  name: str            # "+Z", ...
  letter: str          # "M", ...
  color: str
  rgb: tuple[int, int, int]
  marker_x: np.ndarray  # cube frame
  marker_y: np.ndarray
  normal: np.ndarray    # marker_x x marker_y, outward
  face_center: np.ndarray        # half * normal
  marker_center: np.ndarray      # face_center + offset along marker axes
  corners: np.ndarray            # (4, 3) cube frame, OpenCV order TL, TR, BR, BL

  @property
  def R_cube_marker(self) -> np.ndarray:
    """Rotation taking marker-frame vectors into the cube frame (columns x, y, n)."""
    return np.stack([self.marker_x, self.marker_y, self.normal], axis=1)


@dataclass(frozen=True)
class PrintLayout:
  offset_m: np.ndarray  # (2,) marker-centre offset in marker axes (all faces, unless per_face_m)
  status: str
  source: str
  uncertainty_m: float
  per_face_m: dict[int, np.ndarray] | None = None  # optional per-marker-id offsets (override offset_m)

  @property
  def centered(self) -> bool:
    return bool(np.allclose(self.offset_m, 0.0)) and not self.per_face_m

  def offset_for(self, marker_id: int) -> np.ndarray:
    if self.per_face_m and int(marker_id) in self.per_face_m:
      return self.per_face_m[int(marker_id)]
    return self.offset_m


@dataclass(frozen=True)
class CubeMarkerModel:
  name: str
  cube_size_m: float
  marker_size_m: float
  mass_kg: float
  dictionary_name: str
  body_frame: str
  faces: dict[int, FaceSpec]
  print_layout: PrintLayout
  layout_path: Path
  print_layout_path: Path | None
  warnings: tuple[str, ...] = field(default_factory=tuple)

  @property
  def half_m(self) -> float:
    return 0.5 * self.cube_size_m

  @property
  def marker_ids(self) -> tuple[int, ...]:
    return tuple(sorted(self.faces))

  def dictionary_id(self) -> int:
    import cv2

    return int(getattr(cv2.aruco, self.dictionary_name))

  def object_points(self, ids: Iterable[int]) -> np.ndarray:
    """Stacked (4N, 3) cube-frame corners for marker ``ids`` (unknown IDs raise)."""
    return np.concatenate([self.faces[int(i)].corners for i in ids], axis=0)

  def cube_corners(self) -> np.ndarray:
    h = self.half_m
    return np.array([[sx * h, sy * h, sz * h] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])

  def describe(self) -> dict:
    return {
      "layout_file": str(self.layout_path),
      "print_layout_file": None if self.print_layout_path is None else str(self.print_layout_path),
      "cube_size_m": self.cube_size_m, "half_m": self.half_m, "marker_size_m": self.marker_size_m,
      "mass_kg": self.mass_kg, "dictionary": self.dictionary_name, "body_frame": self.body_frame,
      "marker_offset_in_marker_axes_m": self.print_layout.offset_m.tolist(),
      "print_layout_status": self.print_layout.status,
      "faces": {i: {"face": f.name, "letter": f.letter, "color": f.color, "normal": f.normal.tolist(),
                    "marker_x": f.marker_x.tolist(), "marker_y": f.marker_y.tolist(),
                    "marker_center_m": np.round(f.marker_center, 6).tolist()} for i, f in self.faces.items()},
    }


def _unit(v, what: str) -> np.ndarray:
  a = np.asarray(v, dtype=np.float64)
  if a.shape != (3,) or not np.all(np.isfinite(a)) or not np.isclose(np.linalg.norm(a), 1.0, atol=1e-9):
    raise LayoutError(f"{what} must be a unit 3-vector, got {v}")
  return a


def load_print_layout(path: Path | None = PRINT_LAYOUT_PATH, *, centered: bool = False) -> tuple[PrintLayout, Path | None]:
  """In-face marker placement.  ``centered=True`` forces the (hypothetical) centred layout."""
  if centered:
    return PrintLayout(np.zeros(2), "centered_hypothesis", "forced by caller (comparison only)", 0.0), None
  if path is None or not Path(path).is_file():
    raise LayoutError(f"marker print layout not found: {path} (the marker position inside a face is required)")
  d = json.loads(Path(path).read_text(encoding="utf-8"))
  off = np.asarray(d.get("marker_center_offset_in_marker_axes_m"), dtype=np.float64)
  if off.shape != (2,) or not np.all(np.isfinite(off)):
    raise LayoutError("marker_center_offset_in_marker_axes_m must be two finite numbers")
  status = str(d.get("status", ""))
  allowed = d.get("allowed_status_values")
  if allowed and status not in allowed:
    raise LayoutError(f"unknown print-layout status {status!r}; allowed {allowed}")
  per = None
  if d.get("per_face_offset_in_marker_axes_m"):
    per = {}
    for k, v in d["per_face_offset_in_marker_axes_m"].items():
      a = np.asarray(v, dtype=np.float64)
      if a.shape != (2,) or not np.all(np.isfinite(a)):
        raise LayoutError(f"per-face offset for marker {k} must be two finite numbers")
      per[int(k)] = a
  return PrintLayout(off, status, str(d.get("source", "")), float(d.get("uncertainty_m", np.nan)), per), Path(path)


def load_cube_model(layout_path: Path = LAYOUT_PATH, print_layout_path: Path | None = PRINT_LAYOUT_PATH,
                    *, centered_markers: bool = False, offset_override_m=None) -> CubeMarkerModel:
  """Load and validate the cube marker model.

  ``centered_markers`` / ``offset_override_m`` replace the print layout for analyses
  (layout hypothesis tests); the result is labelled accordingly and never written back.
  """
  d = json.loads(Path(layout_path).read_text(encoding="utf-8"))
  try:
    size = float(d["cube_size_m"])
    msize = float(d["marker_size_m"])
    dictionary = str(d["dictionary"])
    faces_raw = d["faces"]
  except KeyError as exc:
    raise LayoutError(f"layout.json is missing {exc}") from exc
  if not (0 < msize < size):
    raise LayoutError("marker_size_m must be positive and smaller than cube_size_m")
  if offset_override_m is not None:
    if isinstance(offset_override_m, dict):
      per = {int(k): np.asarray(v, dtype=np.float64).reshape(2) for k, v in offset_override_m.items()}
      pl, pl_path = PrintLayout(np.zeros(2), "analysis_override", "set by caller for a layout test", 0.0, per), None
    else:
      off = np.asarray(offset_override_m, dtype=np.float64).reshape(2)
      pl, pl_path = PrintLayout(off, "analysis_override", "set by caller for a layout test", 0.0), None
  else:
    pl, pl_path = load_print_layout(print_layout_path, centered=centered_markers)
  half = 0.5 * size
  faces: dict[int, FaceSpec] = {}
  seen_normals = []
  for f in faces_raw:
    mid = int(f["id"])
    if mid in faces:
      raise LayoutError(f"duplicate marker id {mid}")
    mx = _unit(f["marker_x_in_cube"], f"face {mid} marker_x_in_cube")
    my = _unit(f["marker_y_in_cube"], f"face {mid} marker_y_in_cube")
    if abs(float(mx @ my)) > 1e-9:
      raise LayoutError(f"face {mid}: marker axes are not orthogonal")
    n = np.cross(mx, my)
    name = str(f["name"])
    if name not in _AXIS_NAMES or not np.allclose(n, _AXIS_NAMES[name]):
      raise LayoutError(f"face {mid} ({name}): marker_x x marker_y = {n.tolist()} does not match the face name")
    seen_normals.append(name)
    fc = half * n
    off = pl.offset_for(mid)
    if np.any(np.abs(off) + 0.5 * msize > half + 1e-12):
      raise LayoutError(f"marker {mid}: offset {off.tolist()} puts the marker outside its face")
    mc = fc + off[0] * mx + off[1] * my
    corners = np.array([mc + msize * (u * mx + v * my) for u, v in CORNERS_UNIT])
    faces[mid] = FaceSpec(mid, name, str(f.get("letter", "")), str(f.get("color", "")),
                          tuple(int(c) for c in f.get("rgb", (128, 128, 128))), mx, my, n, fc, mc, corners)
  if sorted(seen_normals) != sorted(_AXIS_NAMES):
    raise LayoutError(f"faces must cover all six axes exactly once, got {seen_normals}")
  return CubeMarkerModel(str(d.get("name", "")), size, msize, float(d.get("mass_kg", np.nan)), dictionary,
                         str(d.get("body_frame", "")), dict(sorted(faces.items())), pl, Path(layout_path), pl_path)


# ----------------------------------------------------------------------------- rotation helpers
def rotvec_to_matrix(rvec) -> np.ndarray:
  r = np.asarray(rvec, dtype=np.float64).reshape(3)
  th = float(np.linalg.norm(r))
  if th < 1e-12:
    return np.eye(3)
  k = r / th
  K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
  return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
  R = np.asarray(R, dtype=np.float64)
  c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
  th = float(np.arccos(c))
  v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
  if th < 1e-7:
    return 0.5 * v
  if np.pi - th < 1e-4:  # near pi: axis from the symmetric part
    M = (R + np.eye(3)) / 2.0
    k = M[:, int(np.argmax(np.diag(M)))]
    k = k / np.linalg.norm(k)
    if k @ v < 0:
      k = -k
    return th * k
  return th / (2.0 * np.sin(th)) * v


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
  c = np.clip((np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1.0) / 2.0, -1.0, 1.0)
  return float(np.degrees(np.arccos(c)))


def project_to_so3(M: np.ndarray) -> np.ndarray:
  U, _, Vt = np.linalg.svd(np.asarray(M, dtype=np.float64))
  R = U @ Vt
  if np.linalg.det(R) < 0:
    U[:, -1] *= -1
    R = U @ Vt
  return R


def quat_wxyz_from_matrix(R: np.ndarray) -> np.ndarray:
  """Same convention as bridge_aruco_pose / cube_pose_io (active rotation, w >= 0)."""
  try:
    from ..bridge_aruco_pose import quaternion_wxyz_from_rotation_matrix
  except ImportError:  # pragma: no cover - script use
    from bridge_aruco_pose import quaternion_wxyz_from_rotation_matrix
  return quaternion_wxyz_from_rotation_matrix(project_to_so3(R))


def matrix_from_quat_wxyz(q) -> np.ndarray:
  try:
    from ..cube_pose_io import rotation_matrix_from_quaternion_wxyz
  except ImportError:  # pragma: no cover
    from cube_pose_io import rotation_matrix_from_quaternion_wxyz
  return rotation_matrix_from_quaternion_wxyz(np.asarray(q, dtype=np.float64))


def cube_symmetry_rotations() -> np.ndarray:
  """The 24 proper rotations mapping the cube onto itself (signed permutation matrices, det +1)."""
  import itertools

  out = []
  for perm in itertools.permutations(range(3)):
    for signs in itertools.product((1, -1), repeat=3):
      M = np.zeros((3, 3))
      for r, (c, s) in enumerate(zip(perm, signs)):
        M[r, c] = s
      if np.isclose(np.linalg.det(M), 1.0):
        out.append(M)
  return np.stack(out)
