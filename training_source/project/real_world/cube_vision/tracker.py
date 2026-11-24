"""Multi-face ArUco cube tracker: one coherent cube pose from every visible marker.

Pipeline per frame
------------------
1. ``cv2.aruco.ArucoDetector`` (DICT_4X4_50) with sub-pixel corner refinement.
2. Association: IDs in layout.json -> faces; any other ID is ignored and reported.
   Corner-order check: OpenCV returns TL, TR, BR, BL clockwise in the image; a quad with a
   non-positive (mirrored) signed area is rejected.
3. Hypotheses: for every visible marker the two IPPE-square solutions of its own 4 corners,
   converted to a *cube* pose through the marker->cube transform of layout.json (+ print
   layout); plus the previous cube pose (temporal continuity).
4. Each hypothesis is scored on ALL visible corners; per-marker reprojection errors decide
   inliers (marker-level RANSAC, exhaustive because there are at most 6 markers).
5. Levenberg-Marquardt refinement (``cv2.solvePnPRefineLM``) on all inlier corners.  The
   cube origin is the cube centre, so tvec IS the cube-centre position in the camera frame.
6. Uncertainty from the PnP Jacobian (sigma^2 (J^T J)^-1); heuristic confidence in [0, 1].

A single visible face gives the classic two-fold IPPE ambiguity.  It is resolved by the
previous pose when that is recent and clearly closer to one solution; otherwise the pose is
reported as ambiguous (``tracking_valid = False``) instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any

import numpy as np

from .geometry import CubeMarkerModel, matrix_to_rotvec, rotation_angle_deg, rotvec_to_matrix
from .intrinsics import CameraIntrinsics, IntrinsicsError


class TrackState(str, Enum):
  TRACKING = "TRACKING"   # >= 2 faces, or 1 face resolved by temporal continuity with a good fit
  PARTIAL = "PARTIAL"     # one face (ambiguous or not), or inlier rejection / poor fit
  LOST = "LOST"           # no reliable marker


@dataclass
class MarkerDetection:
  marker_id: int
  corners: np.ndarray  # (4, 2) TL, TR, BR, BL
  known: bool

  @property
  def center(self) -> np.ndarray:
    return self.corners.mean(axis=0)

  @property
  def side_px(self) -> float:
    c = self.corners
    return float(np.mean([np.linalg.norm(c[k] - c[(k + 1) % 4]) for k in range(4)]))

  @property
  def signed_area(self) -> float:
    x, y = self.corners[:, 0], self.corners[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


@dataclass
class CubePose:
  R: np.ndarray            # rotation camera <- cube
  t: np.ndarray            # cube centre in the camera frame [m]
  inlier_ids: tuple[int, ...]
  rejected_ids: tuple[int, ...]
  reproj_rms_px: float
  reproj_max_px: float
  per_marker_rms_px: dict[int, float]
  method: str
  ambiguous: bool = False
  ambiguity_ratio: float | None = None       # err(second) / err(best), single face only
  alt_rotation_deg: float | None = None      # angle between the two IPPE solutions
  disambiguation: str | None = None          # "temporal" / "reprojection" / None
  pos_std_m: np.ndarray | None = None
  rot_std_deg: np.ndarray | None = None
  confidence: float = 0.0
  layout_conflict_ids: tuple[int, ...] = ()  # rejected markers that fit well alone: the model geometry disagrees

  @property
  def rvec(self) -> np.ndarray:
    return matrix_to_rotvec(self.R)


@dataclass
class TrackResult:
  frame: int
  host_time_s: float
  camera_time_s: float | None
  state: TrackState
  detections: list[MarkerDetection]
  pose: CubePose | None
  message: str = ""
  unknown_ids: tuple[int, ...] = ()
  duplicate_ids: tuple[int, ...] = ()
  rejected_quads: tuple[int, ...] = ()
  extra: dict[str, Any] = field(default_factory=dict)

  @property
  def visible_ids(self) -> tuple[int, ...]:
    return tuple(sorted(d.marker_id for d in self.detections if d.known))

  @property
  def tracking_valid(self) -> bool:
    return self.pose is not None and not self.pose.ambiguous and self.state != TrackState.LOST


@dataclass
class TrackerConfig:
  inlier_marker_rms_px: float = 4.0       # marker is an inlier if its RMS error <= max(this, k * best marker error)
  inlier_relative: float = 3.0
  self_fit_px: float = 1.0                # a rejected marker that fits this well ALONE indicates a geometry/layout conflict
  max_good_rms_px: float = 2.5            # above this the state degrades to PARTIAL
  prior_max_age_s: float = 0.5            # previous pose usable for hypotheses / disambiguation
  prior_max_rotation_deg: float = 35.0    # temporal disambiguation: chosen solution within this of the prior
  ambiguity_ratio_threshold: float = 4.0  # single face w/o prior: unambiguous only if err2/err1 >= this
  min_marker_side_px: float = 8.0
  refine_iterations: int = 30
  max_subpix_shift_px: float = 1.5        # cornerSubPix may not move a corner further than this
  corner_sigma_px: float = 0.25           # corner noise of a frontal marker (synthetic + cornerSubPix)
  min_view_cos: float = 0.2               # oblique-face weighting floor


class MarkerDetector:
  """ArUco detection + association with layout.json; needs no intrinsics."""

  def __init__(self, model: CubeMarkerModel, config: "TrackerConfig | None" = None):
    import cv2

    self.cv2 = cv2
    self.model = model
    self.cfg = config or TrackerConfig()
    dictionary = cv2.aruco.getPredefinedDictionary(model.dictionary_id())
    params = cv2.aruco.DetectorParameters()
    # Built-in refinement is disabled: since OpenCV 4.9 its window is capped at
    # relativeCornerRefinmentWinSize x module size (~1 px for our 20-40 px markers).  We run
    # cv2.cornerSubPix ourselves with a window scaled to the marker (see _refine_corners).
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    if hasattr(cv2.aruco, "ArucoDetector"):
      self._detector = cv2.aruco.ArucoDetector(dictionary, params)
      self._detect = self._detector.detectMarkers
    else:  # OpenCV < 4.7
      self._detect = lambda img: cv2.aruco.detectMarkers(img, dictionary, parameters=params)

  # ------------------------------------------------------------------ detection
  def detect(self, image: np.ndarray) -> tuple[list[MarkerDetection], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    gray = image if image.ndim == 2 else self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY)
    corners, ids, _ = self._detect(gray)
    if ids is not None:
      corners = [self._refine_corners(gray, c) for c in corners]
    dets: list[MarkerDetection] = []
    unknown: list[int] = []
    rejected: list[int] = []
    if ids is None:
      return dets, (), (), ()
    known_ids = set(self.model.faces)
    for c, i in zip(corners, np.asarray(ids).reshape(-1)):
      d = MarkerDetection(int(i), np.asarray(c, dtype=np.float64).reshape(4, 2), int(i) in known_ids)
      if not d.known:
        unknown.append(d.marker_id)
        continue
      if d.signed_area <= 0 or d.side_px < self.cfg.min_marker_side_px:
        rejected.append(d.marker_id)
        continue
      dets.append(d)
    # duplicate IDs (a misdetection: each ID exists once on the cube): keep the larger quad
    by_id: dict[int, MarkerDetection] = {}
    dup: list[int] = []
    for d in dets:
      if d.marker_id in by_id:
        dup.append(d.marker_id)
        if abs(d.signed_area) > abs(by_id[d.marker_id].signed_area):
          by_id[d.marker_id] = d
      else:
        by_id[d.marker_id] = d
    return [by_id[k] for k in sorted(by_id)], tuple(sorted(unknown)), tuple(sorted(set(dup))), tuple(sorted(rejected))

  def _refine_corners(self, gray: np.ndarray, c) -> np.ndarray:
    cv2 = self.cv2
    q = np.asarray(c, np.float32).reshape(4, 1, 2)
    side = float(np.mean([np.linalg.norm(q[k, 0] - q[(k + 1) % 4, 0]) for k in range(4)]))
    win = int(np.clip(round(side / 12.0), 2, 5))
    r = cv2.cornerSubPix(gray, q.copy(), (win, win), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 0.001))
    moved = np.linalg.norm(r - q, axis=2).reshape(4)
    r[moved > self.cfg.max_subpix_shift_px] = q[moved > self.cfg.max_subpix_shift_px]  # keep the detector's corner
    return r.reshape(4, 2)


class CubeTracker:
  def __init__(self, model: CubeMarkerModel, intrinsics: CameraIntrinsics, config: TrackerConfig | None = None):
    import cv2

    if intrinsics is None or not intrinsics.metric:
      raise IntrinsicsError("the tracker needs metric intrinsics (calibrated, device or explicitly estimated)")
    self.cv2 = cv2
    self.model = model
    self.intr = intrinsics
    self.cfg = config or TrackerConfig()
    self.detector = MarkerDetector(model, self.cfg)
    s = model.marker_size_m
    self._square = np.array([[-s / 2, s / 2, 0], [s / 2, s / 2, 0], [s / 2, -s / 2, 0], [-s / 2, -s / 2, 0]])
    self.prev: tuple[float, np.ndarray, np.ndarray] | None = None  # (time, R, t)
    self.frame_index = 0

  def detect(self, image: np.ndarray):
    return self.detector.detect(image)

  # ------------------------------------------------------------------ pose helpers
  def _project(self, obj: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    pts, _ = self.cv2.projectPoints(obj, matrix_to_rotvec(R), np.asarray(t, np.float64), self.intr.camera_matrix,
                                    self.intr.dist_coeffs)
    return pts.reshape(-1, 2)

  def _marker_errors(self, dets: list[MarkerDetection], R, t) -> dict[int, float]:
    obj = self.model.object_points([d.marker_id for d in dets])
    proj = self._project(obj, R, t).reshape(len(dets), 4, 2)
    return {d.marker_id: float(np.sqrt(np.mean(np.sum((proj[k] - d.corners) ** 2, axis=1)))) for k, d in enumerate(dets)}

  def _ippe(self, d: MarkerDetection) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Both IPPE-square solutions as cube poses: [(R_cam_cube, t_cam_cube, rms_px)], best first."""
    cv2 = self.cv2
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(self._square, d.corners, self.intr.camera_matrix, self.intr.dist_coeffs,
                                                flags=cv2.SOLVEPNP_IPPE_SQUARE)
    face = self.model.faces[d.marker_id]
    R_cm, p_cm = face.R_cube_marker, face.marker_center  # marker -> cube
    out = []
    for k in range(int(n)):
      R_cam_m = rotvec_to_matrix(rvecs[k])
      t_cam_m = np.asarray(tvecs[k], np.float64).reshape(3)
      R = R_cam_m @ R_cm.T
      t = t_cam_m - R @ p_cm
      err = float(np.asarray(errs).reshape(-1)[k]) if errs is not None else math.nan
      out.append((R, t, err))
    out.sort(key=lambda s: s[2])
    return out

  def _weights(self, dets: list[MarkerDetection], R, t) -> np.ndarray:
    """Per-corner weight 1/sigma^2 with sigma = corner_sigma / max(cos(view angle), floor)."""
    w = []
    for d in dets:
      f = self.model.faces[d.marker_id]
      c = R @ f.marker_center + t
      cosang = float((R @ f.normal) @ (-c) / max(np.linalg.norm(c), 1e-9))
      sig = self.cfg.corner_sigma_px / max(cosang, self.cfg.min_view_cos)
      w += [1.0 / sig ** 2] * 4
    return np.asarray(w)

  def _refine(self, dets: list[MarkerDetection], R, t) -> tuple[np.ndarray, np.ndarray]:
    """LM (OpenCV) followed by obliqueness-weighted Gauss-Newton on all corners of ``dets``."""
    cv2 = self.cv2
    obj = self.model.object_points([d.marker_id for d in dets])
    img = np.concatenate([d.corners for d in dets], axis=0)
    rvec = matrix_to_rotvec(R).reshape(3, 1).copy()
    tvec = np.asarray(t, np.float64).reshape(3, 1).copy()
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, self.cfg.refine_iterations, 1e-10)
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, self.intr.camera_matrix, self.intr.dist_coeffs, rvec, tvec, crit)
    if len(dets) < 2:
      return rotvec_to_matrix(rvec), tvec.reshape(3)
    x = np.concatenate([rvec.reshape(3), tvec.reshape(3)])
    for _ in range(10):
      Rk = rotvec_to_matrix(x[:3])
      w = np.repeat(self._weights(dets, Rk, x[3:]), 2)
      proj, J = cv2.projectPoints(obj, x[:3], x[3:], self.intr.camera_matrix, self.intr.dist_coeffs)
      r = (proj.reshape(-1, 2) - img).reshape(-1)
      J6 = J[:, :6]
      A = J6.T @ (w[:, None] * J6)
      g = J6.T @ (w * r)
      try:
        dx = -np.linalg.solve(A + 1e-9 * np.eye(6), g)
      except np.linalg.LinAlgError:
        break
      x = x + dx
      if np.linalg.norm(dx) < 1e-10:
        break
    return rotvec_to_matrix(x[:3]), x[3:].copy()

  def _uncertainty(self, dets, R, t, rms: float) -> tuple[np.ndarray, np.ndarray]:
    obj = self.model.object_points([d.marker_id for d in dets])
    _, J = self.cv2.projectPoints(obj, matrix_to_rotvec(R), np.asarray(t, np.float64), self.intr.camera_matrix,
                                  self.intr.dist_coeffs)
    J6 = J[:, :6]
    m = J6.shape[0]
    sigma2 = (rms ** 2) * m / max(m - 6, 1) if m > 6 else 0.5 ** 2
    sigma2 = max(sigma2, 0.1 ** 2)  # corner noise floor
    try:
      cov = sigma2 * np.linalg.inv(J6.T @ J6)
    except np.linalg.LinAlgError:
      return np.full(3, np.inf), np.full(3, np.inf)
    d = np.sqrt(np.clip(np.diag(cov), 0, None))
    return d[3:6], np.degrees(d[0:3])

  # ------------------------------------------------------------------ main entry
  def estimate(self, dets: list[MarkerDetection], now_s: float) -> tuple[CubePose | None, str]:
    cfg = self.cfg
    if not dets:
      return None, "no known marker visible"
    prior = None
    if self.prev is not None and now_s - self.prev[0] <= cfg.prior_max_age_s:
      prior = self.prev
    sols = {d.marker_id: self._ippe(d) for d in dets}

    if len(dets) == 1:
      d = dets[0]
      s = sols[d.marker_id]
      best = s[0]
      ratio = (s[1][2] / max(s[0][2], 1e-9)) if len(s) > 1 else math.inf
      alt = rotation_angle_deg(s[0][0], s[1][0]) if len(s) > 1 else 0.0
      how = "reprojection"
      ambiguous = ratio < cfg.ambiguity_ratio_threshold
      if prior is not None and len(s) > 1:
        a0, a1 = rotation_angle_deg(prior[1], s[0][0]), rotation_angle_deg(prior[1], s[1][0])
        k = 0 if a0 <= a1 else 1
        if min(a0, a1) <= cfg.prior_max_rotation_deg and max(a0, a1) > min(a0, a1) + 10.0:
          best, how, ambiguous = s[k], "temporal", False
      R, t = self._refine(dets, best[0], best[1])
      errs = self._marker_errors(dets, R, t)
      rms = errs[d.marker_id]
      ps, rs = self._uncertainty(dets, R, t, rms)
      pose = CubePose(R, t, (d.marker_id,), (), rms, rms, errs, "single_face_ippe+LM", ambiguous, float(ratio), float(alt),
                      how if not ambiguous else None, ps, rs)
      pose.confidence = _confidence(pose, 1)
      return pose, ("single face, ambiguous IPPE solution" if ambiguous else f"single face, resolved by {how}")

    # multi-face: hypotheses from every marker's IPPE solutions (+ prior), scored on all corners
    hyps = [(s[0], s[1]) for v in sols.values() for s in v]
    if prior is not None:
      hyps.append((prior[1], prior[2]))
    best = None
    for R0, t0 in hyps:
      errs = self._marker_errors(dets, R0, t0)
      thr = max(cfg.inlier_marker_rms_px, cfg.inlier_relative * min(errs.values()))
      inl = [d for d in dets if errs[d.marker_id] <= thr]
      score = (len(inl), -float(np.mean([errs[d.marker_id] for d in inl])) if inl else -1e9)
      if best is None or score > best[0]:
        best = (score, R0, t0, inl)
    _, R0, t0, inl = best
    if len(inl) == 0:
      return None, "no consistent pose hypothesis"
    R, t = self._refine(inl, R0, t0)
    errs = self._marker_errors(dets, R, t)
    thr = max(cfg.inlier_marker_rms_px, cfg.inlier_relative * min(errs[d.marker_id] for d in inl))
    inl2 = [d for d in dets if errs[d.marker_id] <= thr]
    if len(inl2) != len(inl) and inl2:
      inl = inl2
      R, t = self._refine(inl, R, t)
      errs = self._marker_errors(dets, R, t)
    rejected = tuple(sorted(d.marker_id for d in dets if d not in inl))
    conflicts = tuple(sorted(i for i in rejected if sols[i][0][2] <= cfg.self_fit_px))
    obj_err = np.array([errs[d.marker_id] for d in inl])
    rms = float(np.sqrt(np.mean(obj_err ** 2)))
    ps, rs = self._uncertainty(inl, R, t, rms)
    if len(inl) == 1:  # all other markers rejected: fall back to single-face semantics
      pose1, msg1 = self.estimate(inl, now_s)
      if pose1 is not None:
        pose1.rejected_ids, pose1.layout_conflict_ids = rejected, conflicts
      return pose1, msg1 + f", rejected {list(rejected)}"
    pose = CubePose(R, t, tuple(sorted(d.marker_id for d in inl)), rejected, rms, float(max(obj_err)), errs,
                    "multi_face_hypothesis_ransac+LM", False, None, None, None, ps, rs)
    pose.layout_conflict_ids = conflicts
    pose.confidence = _confidence(pose, len(inl))
    return pose, f"{len(inl)} faces" + (f", rejected {list(rejected)}" if rejected else "")

  def process(self, image: np.ndarray, host_time_s: float, camera_time_s: float | None = None) -> TrackResult:
    h, w = image.shape[:2]
    self.intr.check_frame(w, h)
    dets, unknown, dup, rejq = self.detect(image)
    pose, msg = self.estimate(dets, host_time_s)
    if pose is None:
      state = TrackState.LOST
    elif len(pose.inlier_ids) >= 2 and pose.reproj_rms_px <= self.cfg.max_good_rms_px:
      state = TrackState.TRACKING
    elif len(pose.inlier_ids) == 1 and not pose.ambiguous and pose.disambiguation == "temporal" \
        and pose.reproj_rms_px <= self.cfg.max_good_rms_px:
      state = TrackState.TRACKING
    else:
      state = TrackState.PARTIAL
    res = TrackResult(self.frame_index, host_time_s, camera_time_s, state, dets, pose, msg, unknown, dup, rejq)
    if pose is not None and not pose.ambiguous:
      self.prev = (host_time_s, pose.R.copy(), pose.t.copy())
    self.frame_index += 1
    return res


def _confidence(p: CubePose, n_faces: int) -> float:
  """Heuristic in [0, 1] (documented, not a probability): fit quality x face count x ambiguity."""
  fit = math.exp(-max(p.reproj_rms_px, 0.0) / 2.0)
  faces = {1: 0.55, 2: 0.85}.get(n_faces, 1.0)
  amb = 0.3 if p.ambiguous else 1.0
  rej = 0.8 if p.rejected_ids else 1.0
  return float(np.clip(fit * faces * amb * rej, 0.0, 1.0))
