"""Metric depth for the tracked cube: measured RGB-D depth, fused with the marker PnP pose.

Labels (never mixed up):
  RGB_D_FUSED        a depth stream aligned to the colour image was sampled inside the visible
                     markers, passed the consistency checks, and was fused with the PnP range
  PNP_METRIC         no usable depth stream (or it failed the checks): the range comes from the
                     calibrated camera geometry and the known 15.55 mm marker side (PnP)
  DEPTH_UNAVAILABLE  no metric result (no pose, or no metric intrinsics)

Fusion model: the image position of the cube is precise; the uncertain quantity is the range
along the viewing ray.  The depth stream gives, per visible marker, the median depth of
pixels inside the marker (shrunk quad, away from the black/white border), deprojected to a
3D marker centre; with the PnP orientation this gives a depth-derived cube centre.  The range
of the cube centre (|p|) is then an inverse-variance mean of the PnP range and the depth
range, and the fused position is the PnP position rescaled along its ray.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .intrinsics import CameraIntrinsics


class DepthSource(str, Enum):
  RGB_D_FUSED = "RGB_D_FUSED"
  PNP_METRIC = "PNP_METRIC"
  DEPTH_UNAVAILABLE = "DEPTH_UNAVAILABLE"


@dataclass
class DepthConfig:
  shrink: float = 0.6                 # sample inside the marker quad scaled by this about its centre
  min_valid_ratio: float = 0.5        # share of sampled pixels with valid depth
  min_valid_pixels: int = 12
  min_depth_m: float = 0.10           # valid sensor range (RealSense D4xx ~0.1-3 m)
  max_depth_m: float = 2.0
  max_mad_m: float = 0.004            # flat marker: robust spread must be small
  max_range_diff_m: float = 0.010     # |depth range - PnP range| allowed: max(this, rel * range)
  max_range_diff_rel: float = 0.03
  max_lateral_diff_m: float = 0.010   # depth-derived centre vs PnP centre, perpendicular to the ray
  depth_noise_floor_m: float = 0.0015  # per-marker sigma floor (sensor quantisation / bias)


@dataclass
class MarkerDepth:
  marker_id: int
  n_sampled: int
  n_valid: int
  median_m: float | None
  mad_m: float | None
  center_cam_m: np.ndarray | None     # deprojected marker centre (depth-derived)
  cube_center_cam_m: np.ndarray | None
  ok: bool
  reason: str = ""


@dataclass
class DepthResult:
  source: DepthSource
  position_m: np.ndarray | None        # final cube-centre estimate in the camera frame
  range_m: float | None                # |position|
  z_m: float | None                    # camera-frame Z (optical axis depth) of the cube centre
  sigma_range_m: float | None
  pnp_range_m: float | None
  pnp_sigma_range_m: float | None
  depth_range_m: float | None          # from the depth stream (None if not measured)
  depth_sigma_range_m: float | None
  valid_ratio: float | None            # valid / sampled depth pixels over the visible markers
  n_valid: int = 0
  n_sampled: int = 0
  confidence: float = 0.0
  status: str = ""                     # why depth was / was not fused
  markers: list[MarkerDepth] = field(default_factory=list)


def quad_mask(shape_hw, quad: np.ndarray, shrink: float) -> np.ndarray:
  import cv2

  c = quad.mean(axis=0)
  q = c + shrink * (quad - c)
  m = np.zeros(shape_hw, np.uint8)
  cv2.fillConvexPoly(m, np.round(q).astype(np.int32), 1)
  return m.astype(bool)


def deproject(u: np.ndarray, v: np.ndarray, z: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
  """Pixel (u, v) + depth z (along the optical axis) -> camera-frame points; undistorts if needed."""
  import cv2

  pts = np.stack([u, v], axis=-1).astype(np.float64).reshape(-1, 1, 2)
  if np.any(intr.dist_coeffs != 0):
    n = cv2.undistortPoints(pts, intr.camera_matrix, intr.dist_coeffs).reshape(-1, 2)
  else:
    n = np.stack([(pts[:, 0, 0] - intr.cx) / intr.fx, (pts[:, 0, 1] - intr.cy) / intr.fy], axis=-1)
  z = np.asarray(z, np.float64).reshape(-1)
  return np.stack([n[:, 0] * z, n[:, 1] * z, z], axis=-1)


def depth_to_meters(depth_raw: np.ndarray, scale_m_per_unit: float | None) -> np.ndarray:
  """uint16 sensor units -> metres; float arrays are taken as metres already.  0 / non-finite -> NaN."""
  d = np.asarray(depth_raw)
  if np.issubdtype(d.dtype, np.integer):
    if scale_m_per_unit is None or not np.isfinite(scale_m_per_unit) or scale_m_per_unit <= 0:
      raise ValueError("integer depth needs a positive depth_scale_m_per_unit")
    out = d.astype(np.float64) * float(scale_m_per_unit)
    out[d == 0] = np.nan
  else:
    out = d.astype(np.float64)
    out[~np.isfinite(out) | (out <= 0)] = np.nan
  return out


def estimate_depth(track, depth_m: np.ndarray | None, intr: CameraIntrinsics, model, cfg: DepthConfig | None = None) -> DepthResult:
  """``track``: a TrackResult; ``depth_m``: aligned depth in metres (NaN = invalid) or None;
  ``model``: the CubeMarkerModel used by the tracker."""
  cfg = cfg or DepthConfig()
  pose = getattr(track, "pose", None)
  if pose is None or intr is None or not intr.metric:
    return DepthResult(DepthSource.DEPTH_UNAVAILABLE, None, None, None, None, None, None, None, None, None,
                       status="no pose" if pose is None else "no metric intrinsics")
  p = np.asarray(pose.t, np.float64)
  r_pnp = float(np.linalg.norm(p))
  ray = p / r_pnp
  # PnP range sigma from the pose covariance, projected on the ray (+ a floor)
  sp = pose.pos_std_m if pose.pos_std_m is not None and np.all(np.isfinite(pose.pos_std_m)) else np.full(3, 0.01)
  sig_pnp = float(max(np.sqrt(np.sum((ray * sp) ** 2)), 5e-4))
  base = DepthResult(DepthSource.PNP_METRIC, p.copy(), r_pnp, float(p[2]), sig_pnp, r_pnp, sig_pnp, None, None, None,
                     confidence=float(pose.confidence))
  if depth_m is None:
    base.status = "no depth stream"
    return base
  if depth_m.shape[:2] != (intr.image_size_wh[1], intr.image_size_wh[0]):
    base.status = f"depth {depth_m.shape[1]}x{depth_m.shape[0]} is not aligned to the colour image"
    return base
  model_faces = model.faces
  markers: list[MarkerDepth] = []
  n_s = n_v = 0
  for d in track.detections:
    if d.marker_id not in pose.inlier_ids:
      continue
    m = quad_mask(depth_m.shape, d.corners, cfg.shrink)
    vv, uu = np.nonzero(m)
    z = depth_m[vv, uu]
    ok_px = np.isfinite(z) & (z >= cfg.min_depth_m) & (z <= cfg.max_depth_m)
    n_s += len(z)
    n_v += int(ok_px.sum())
    md = MarkerDepth(d.marker_id, len(z), int(ok_px.sum()), None, None, None, None, False)
    if ok_px.sum() < cfg.min_valid_pixels or ok_px.mean() < cfg.min_valid_ratio:
      md.reason = f"valid depth {ok_px.sum()}/{len(z)}"
      markers.append(md)
      continue
    pts = deproject(uu[ok_px], vv[ok_px], z[ok_px], intr)
    zmed = float(np.median(z[ok_px]))
    mad = float(1.4826 * np.median(np.abs(z[ok_px] - zmed)))
    md.median_m, md.mad_m = zmed, mad
    if mad > cfg.max_mad_m:
      md.reason = f"depth spread {1000 * mad:.1f} mm (mixed pixels / not flat)"
      markers.append(md)
      continue
    c3 = np.median(pts, axis=0)
    md.center_cam_m = c3
    mc = model_faces[d.marker_id].marker_center
    md.cube_center_cam_m = c3 - pose.R @ mc
    md.ok = True
    markers.append(md)
  base.markers = markers
  base.n_sampled, base.n_valid = n_s, n_v
  base.valid_ratio = (n_v / n_s) if n_s else 0.0
  good = [m for m in markers if m.ok]
  if not good:
    base.status = "depth rejected: " + "; ".join(f"id {m.marker_id}: {m.reason}" for m in markers) if markers \
      else "no inlier marker to sample"
    return base
  # combine markers (weights from per-marker spread / sample count)
  sig = np.array([max(m.mad_m / np.sqrt(max(m.n_valid, 1)), cfg.depth_noise_floor_m) for m in good])
  w = 1 / sig ** 2
  c_depth = np.sum(np.stack([m.cube_center_cam_m for m in good]) * w[:, None], axis=0) / w.sum()
  sig_d = float(np.sqrt(1 / w.sum()))
  r_d = float(np.linalg.norm(c_depth))
  base.depth_range_m, base.depth_sigma_range_m = r_d, sig_d
  lateral = float(np.linalg.norm(c_depth - (c_depth @ ray) * ray))
  tol = max(cfg.max_range_diff_m, cfg.max_range_diff_rel * r_pnp)
  if abs(r_d - r_pnp) > tol:
    base.status = f"depth rejected: range differs from PnP by {1000 * (r_d - r_pnp):+.1f} mm (limit {1000 * tol:.0f} mm)"
    base.confidence = float(pose.confidence) * 0.8
    return base
  if lateral > cfg.max_lateral_diff_m:
    base.status = f"depth rejected: depth-derived centre is {1000 * lateral:.1f} mm off the PnP ray"
    base.confidence = float(pose.confidence) * 0.8
    return base
  wp, wd = 1 / sig_pnp ** 2, 1 / sig_d ** 2
  r_f = (wp * r_pnp + wd * r_d) / (wp + wd)
  sig_f = float(np.sqrt(1 / (wp + wd)))
  pos = ray * r_f
  base.source = DepthSource.RGB_D_FUSED
  base.position_m, base.range_m, base.z_m, base.sigma_range_m = pos, float(r_f), float(pos[2]), sig_f
  base.confidence = float(min(1.0, pose.confidence * (0.85 + 0.15 * base.valid_ratio)))
  base.status = f"fused {len(good)} marker(s): depth {1000 * (r_d - r_pnp):+.1f} mm vs PnP"
  return base
