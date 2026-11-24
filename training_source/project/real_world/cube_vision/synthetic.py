"""Synthetic RGB(-D) renderer of the MBZUAI cube, for numerical tests and hardware-free demos.

Draws the visible faces (painter's algorithm, far to near) with their layout.json colour, a
white quiet zone and the real DICT_4X4_50 marker warped onto the marker's 3D corners
(from the same ``CubeMarkerModel`` the tracker uses).  The optional depth map is exact
(ray-plane intersection per pixel) and can be degraded with noise / holes.  Zero lens
distortion (warpPerspective cannot render distortion).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import CubeMarkerModel, matrix_to_rotvec, rotvec_to_matrix
from .intrinsics import STATUS_CHECKERBOARD, CameraIntrinsics


def synthetic_intrinsics(width: int = 640, height: int = 480, f: float = 600.0) -> CameraIntrinsics:
  K = np.array([[f, 0, width / 2 - 0.5], [0, f, height / 2 - 0.5], [0, 0, 1.0]])
  return CameraIntrinsics(K, np.zeros(5), (width, height), STATUS_CHECKERBOARD, "synthetic", 0.0,
                          depth_scale_m_per_unit=0.001, depth_aligned_to_color=True)


@dataclass
class SyntheticFrame:
  bgr: np.ndarray
  depth_m: np.ndarray | None
  R: np.ndarray
  t: np.ndarray
  visible_faces: tuple[int, ...]


def look_rotation(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
  return rotvec_to_matrix(np.radians([rx_deg, 0, 0])) @ rotvec_to_matrix(np.radians([0, ry_deg, 0])) @ \
    rotvec_to_matrix(np.radians([0, 0, rz_deg]))


class CubeRenderer:
  def __init__(self, model: CubeMarkerModel, intr: CameraIntrinsics, supersample: int = 2, quiet_zone: float = 1.35,
               background_bgr=(70, 70, 70)):
    import cv2

    self.cv2, self.model, self.intr = cv2, model, intr
    self.ss = int(supersample)
    self.quiet = float(quiet_zone)
    self.bg = background_bgr
    d = cv2.aruco.getPredefinedDictionary(model.dictionary_id())
    self.tex = {i: cv2.cvtColor(cv2.aruco.generateImageMarker(d, i, 240, borderBits=1), cv2.COLOR_GRAY2BGR)
                for i in model.faces}

  def _proj(self, pts_cube: np.ndarray, R, t, scale: float) -> np.ndarray:
    pc = pts_cube @ R.T + t
    K = self.intr.camera_matrix
    uv = (pc[:, :2] / pc[:, 2:3]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    # supersampled pixel centres: (u + 0.5) * s - 0.5
    return (uv + 0.5) * scale - 0.5

  def face_quads(self, R, t):
    """Visible faces sorted far -> near: (id, face square (4,3), marker square (4,3), quiet zone (4,3))."""
    m = self.model
    h, s = m.half_m, m.marker_size_m
    out = []
    for i, f in m.faces.items():
      n_cam = R @ f.normal
      c_cam = R @ f.face_center + t
      if n_cam @ (-c_cam) <= 1e-9:
        continue
      sq = np.array([f.face_center + h * (u * f.marker_x + v * f.marker_y) for u, v in ((-1, 1), (1, 1), (1, -1), (-1, -1))])
      qz = np.array([f.marker_center + self.quiet * s * (u * f.marker_x + v * f.marker_y)
                     for u, v in ((-.5, .5), (.5, .5), (.5, -.5), (-.5, -.5))])
      out.append((float(c_cam[2]), i, sq, f.corners, qz))
    out.sort(key=lambda x: -x[0])
    return [(i, sq, mk, qz) for _, i, sq, mk, qz in out]

  def render(self, R: np.ndarray, t: np.ndarray, *, depth: bool = False, background_depth_m: float | None = None,
             occluders: list[tuple[int, int, int, int]] | None = None, noise_std: float = 0.0, rng=None) -> SyntheticFrame:
    cv2 = self.cv2
    W, H = self.intr.image_size_wh
    s = self.ss
    img = np.empty((H * s, W * s, 3), np.uint8)
    img[:] = self.bg
    faces = self.face_quads(R, t)
    for i, sq, mk, qz in faces:
      f = self.model.faces[i]
      col = tuple(int(c) for c in f.rgb[::-1])
      cv2.fillConvexPoly(img, np.round(self._proj(sq, R, t, s) * 16).astype(np.int32), col, cv2.LINE_AA, shift=4)
      cv2.fillConvexPoly(img, np.round(self._proj(qz, R, t, s) * 16).astype(np.int32), (255, 255, 255), cv2.LINE_AA, shift=4)
      dst = self._proj(mk, R, t, s).astype(np.float32)
      tex = self.tex[i]
      n = tex.shape[0]
      src = np.float32([[-0.5, -0.5], [n - 0.5, -0.5], [n - 0.5, n - 0.5], [-0.5, n - 0.5]])
      Hm = cv2.getPerspectiveTransform(src, dst)
      warped = cv2.warpPerspective(tex, Hm, (W * s, H * s), flags=cv2.INTER_LINEAR)
      mask = cv2.warpPerspective(np.full((n, n), 255, np.uint8), Hm, (W * s, H * s), flags=cv2.INTER_LINEAR)
      a = (mask.astype(np.float32) / 255.0)[..., None]
      img = (img * (1 - a) + warped * a).astype(np.uint8)
    if occluders:
      for x0, y0, x1, y1 in occluders:
        cv2.rectangle(img, (x0 * s, y0 * s), (x1 * s, y1 * s), (40, 40, 40), -1)
    out = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA) if s > 1 else img
    if noise_std > 0:
      rng = rng or np.random.default_rng(0)
      out = np.clip(out.astype(np.float32) + rng.normal(0, noise_std, out.shape), 0, 255).astype(np.uint8)
    dmap = self.render_depth(R, t, faces, background_depth_m, occluders) if depth else None
    return SyntheticFrame(out, dmap, R, t, tuple(i for i, *_ in faces))

  def render_depth(self, R, t, faces, background_depth_m, occluders) -> np.ndarray:
    cv2 = self.cv2
    W, H = self.intr.image_size_wh
    K = self.intr.camera_matrix
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1)
    depth = np.full((H, W), 0.0 if background_depth_m is None else background_depth_m)
    for i, sq, *_ in faces:
      f = self.model.faces[i]
      n = R @ f.normal
      p0 = R @ f.face_center + t
      mask = np.zeros((H, W), np.uint8)
      cv2.fillConvexPoly(mask, np.round(self._proj(sq, R, t, 1.0)).astype(np.int32), 1)
      z = (n @ p0) / (rays @ n)
      depth = np.where(mask > 0, z, depth)
    if occluders:
      for x0, y0, x1, y1 in occluders:
        depth[y0:y1, x0:x1] = np.minimum(depth[y0:y1, x0:x1], 0.25) if background_depth_m else 0.25
    return depth


def degrade_depth(depth_m: np.ndarray, *, noise_std_m: float = 0.0, hole_fraction: float = 0.0, scale_m_per_unit=0.001,
                  rng=None) -> np.ndarray:
  """Quantise to uint16 depth units (RealSense-like), add noise and zero-valued holes."""
  rng = rng or np.random.default_rng(1)
  d = depth_m + (rng.normal(0, noise_std_m, depth_m.shape) if noise_std_m > 0 else 0.0)
  units = np.clip(np.round(d / scale_m_per_unit), 0, 65535).astype(np.uint16)
  if hole_fraction > 0:
    units[rng.random(units.shape) < hole_fraction] = 0
  units[depth_m <= 0] = 0
  return units


def random_pose(rng, distance_m=(0.25, 0.45), max_offset_m=0.04) -> tuple[np.ndarray, np.ndarray]:
  rv = rng.normal(size=3)
  rv = rv / np.linalg.norm(rv) * rng.uniform(0, np.pi)
  R = rotvec_to_matrix(rv)
  t = np.array([rng.uniform(-max_offset_m, max_offset_m), rng.uniform(-max_offset_m, max_offset_m), rng.uniform(*distance_m)])
  return R, t


__all__ = ["CubeRenderer", "SyntheticFrame", "synthetic_intrinsics", "degrade_depth", "random_pose", "look_rotation",
           "matrix_to_rotvec"]
