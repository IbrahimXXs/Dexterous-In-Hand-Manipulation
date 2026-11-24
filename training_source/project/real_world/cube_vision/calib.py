"""Calibration utilities: ChArUco/chessboard intrinsics, camera-to-palm capture + solve,
recording inventory, synthetic demo recording."""

from __future__ import annotations

import datetime as _dt
import glob
import json
from pathlib import Path
import time

import numpy as np

from .intrinsics import STATUS_CHECKERBOARD, STATUS_DEVICE_VERIFIED, STATUS_UNCALIBRATED, CameraIntrinsics, load_intrinsics

# Board dictionary deliberately different from the cube's DICT_4X4_50, so board markers can
# never be mistaken for cube markers (and vice versa).
DEFAULT_BOARD = {"type": "charuco", "squares_x": 7, "squares_y": 5, "square_m": 0.030, "marker_m": 0.022,
                 "dictionary": "DICT_5X5_100"}


def _board(spec: dict):
  import cv2

  d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec["dictionary"]))
  return cv2.aruco.CharucoBoard((spec["squares_x"], spec["squares_y"]), spec["square_m"], spec["marker_m"], d), d


def make_board(out_png: Path, spec: dict | None = None, dpi: int = 300) -> Path:
  import cv2

  spec = dict(spec or DEFAULT_BOARD)
  board, _ = _board(spec)
  px_per_m = dpi / 0.0254
  w = int(round(spec["squares_x"] * spec["square_m"] * px_per_m))
  h = int(round(spec["squares_y"] * spec["square_m"] * px_per_m))
  img = board.generateImage((w, h), marginSize=0)
  cv2.imwrite(str(out_png), img)
  spec.update({"dpi": dpi, "print_note": "Print at 100% (no scaling); measure one square with calipers and pass "
               "--square-m if it differs."})
  Path(out_png).with_suffix(".json").write_text(json.dumps(spec, indent=2) + "\n")
  return out_png


def _detect_charuco(gray, spec):
  import cv2

  board, _ = _board(spec)
  det = cv2.aruco.CharucoDetector(board)
  cc, ci, _, _ = det.detectBoard(gray)
  if ci is None or len(ci) < 6:
    return None
  obj, img = board.matchImagePoints(cc, ci)
  return obj.reshape(-1, 3), img.reshape(-1, 2)


def _detect_chessboard(gray, spec):
  import cv2

  ok, c = cv2.findChessboardCorners(gray, (spec["cols"], spec["rows"]))
  if not ok:
    return None
  c = cv2.cornerSubPix(gray, c, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 50, 1e-4))
  objp = np.zeros((spec["rows"] * spec["cols"], 3))
  objp[:, :2] = np.mgrid[0:spec["cols"], 0:spec["rows"]].T.reshape(-1, 2) * spec["square_m"]
  return objp, c.reshape(-1, 2)


def calibrate_intrinsics(images: list[np.ndarray], spec: dict, camera_id: str, *, max_rms_px: float = 1.0,
                         min_views: int = 12) -> tuple[CameraIntrinsics, dict]:
  import cv2

  obj_pts, img_pts, size = [], [], None
  for im in images:
    g = im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    size = (g.shape[1], g.shape[0])
    r = _detect_charuco(g, spec) if spec["type"] == "charuco" else _detect_chessboard(g, spec)
    if r is not None:
      obj_pts.append(r[0].astype(np.float32))
      img_pts.append(r[1].astype(np.float32))
  report = {"views_offered": len(images), "views_used": len(obj_pts), "board": spec}
  if len(obj_pts) < 3:
    raise ValueError(f"only {len(obj_pts)} usable board views (need >= {min_views})")
  rms, K, dist, rv, tv = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
  per = []
  for o, i, r, t in zip(obj_pts, img_pts, rv, tv):
    p, _ = cv2.projectPoints(o, r, t, K, dist)
    per.append(float(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - i) ** 2, axis=1)))))
  ok = rms <= max_rms_px and len(obj_pts) >= min_views
  report.update({"rms_px": float(rms), "per_view_rms_px": per, "accepted": bool(ok),
                 "acceptance": {"max_rms_px": max_rms_px, "min_views": min_views}})
  intr = CameraIntrinsics(K, dist.reshape(-1), size, STATUS_CHECKERBOARD if ok else STATUS_UNCALIBRATED, camera_id, float(rms),
                          extra={"n_views": len(obj_pts), "board": spec, "per_view_rms_px": per,
                                 "acceptance": report["acceptance"]})
  return intr, report


def collect_images(images_glob: str | None, video: str | None, every: int = 15) -> list[np.ndarray]:
  import cv2

  out = []
  if images_glob:
    for p in sorted(glob.glob(images_glob)):
      im = cv2.imread(p)
      if im is not None:
        out.append(im)
  if video:
    cap = cv2.VideoCapture(video)
    i = 0
    while True:
      ok, im = cap.read()
      if not ok:
        break
      if i % every == 0:
        out.append(im)
      i += 1
  return out


def capture_board_views(source, spec: dict, n_views: int = 25, min_interval_s: float = 0.7) -> list[np.ndarray]:
  """Interactive: shows the camera, highlights board corners; SPACE captures, q finishes."""
  import cv2

  views = []
  last = 0.0
  while len(views) < n_views:
    f = source.read()
    if f is None:
      break
    g = cv2.cvtColor(f.bgr, cv2.COLOR_BGR2GRAY)
    r = _detect_charuco(g, spec) if spec["type"] == "charuco" else _detect_chessboard(g, spec)
    vis = f.bgr.copy()
    if r is not None:
      for u, v in r[1]:
        cv2.circle(vis, (int(u), int(v)), 3, (0, 255, 0), -1)
    cv2.putText(vis, f"views {len(views)}/{n_views}  SPACE capture  q done  (tilt/move the board, cover the corners)",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.imshow("calibrate intrinsics", vis)
    k = cv2.waitKey(1)
    if k in (ord("q"), 27):
      break
    if k == ord(" ") and r is not None and time.time() - last > min_interval_s:
      views.append(f.bgr.copy())
      last = time.time()
  cv2.destroyAllWindows()
  return views


def compare_intrinsics(a: CameraIntrinsics, b: CameraIntrinsics) -> dict:
  """Pixel displacement between two models over the image (e.g. factory vs board calibration)."""
  import cv2

  w, h = a.image_size_wh
  u, v = np.meshgrid(np.linspace(0, w - 1, 25), np.linspace(0, h - 1, 19))
  pts = np.stack([u.ravel(), v.ravel()], -1).reshape(-1, 1, 2).astype(np.float64)
  n = cv2.undistortPoints(pts, a.camera_matrix, a.dist_coeffs)
  rays = np.concatenate([n.reshape(-1, 2), np.ones((len(n), 1))], 1)
  p, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), b.camera_matrix, b.dist_coeffs)
  d = np.linalg.norm(p.reshape(-1, 2) - pts.reshape(-1, 2), axis=1)
  return {"max_px": float(d.max()), "median_px": float(np.median(d)), "fx_rel_diff": float(b.fx / a.fx - 1),
          "fy_rel_diff": float(b.fy / a.fy - 1)}


def verify_device_intrinsics(device: CameraIntrinsics, board_cal: CameraIntrinsics, max_px: float = 1.0) -> tuple[CameraIntrinsics, dict]:
  cmp = compare_intrinsics(device, board_cal)
  ok = cmp["max_px"] <= max_px and board_cal.status == STATUS_CHECKERBOARD
  out = CameraIntrinsics(device.camera_matrix, device.dist_coeffs, device.image_size_wh,
                         STATUS_DEVICE_VERIFIED if ok else device.status, device.camera_id, board_cal.rms_reprojection_px,
                         device.depth_scale_m_per_unit, device.depth_aligned_to_color,
                         extra={"verified_against": board_cal.source_path, "comparison": cmp, "max_px_allowed": max_px})
  return out, cmp


# ----------------------------------------------------------------------------- camera-to-palm
def capture_reference_poses(source, pipeline, reference: list[dict], frames_per_placement: int = 60,
                            min_faces: int = 2) -> list[dict]:
  """Interactive: for each placement, place the cube in the jig, press SPACE; averages
  ``frames_per_placement`` TRACKING frames with >= ``min_faces`` faces."""
  import cv2

  from .dashboard import annotate_camera
  from .extrinsics import average_pose

  caps = []
  for ref in reference:
    print(f"[camera-to-palm] place the cube at '{ref['name']}' and press SPACE (q aborts)")
    collecting, Rs, ts = False, [], []
    while True:
      f = source.read()
      if f is None:
        raise RuntimeError("camera stopped")
      st = pipeline.process(f)
      vis = annotate_camera(f.bgr, st, pipeline.model, pipeline.intr)
      cv2.putText(vis, f"{ref['name']}: {'collecting %d' % len(Rs) if collecting else 'SPACE to collect'}", (10, 470),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
      cv2.imshow("camera-to-palm capture", vis)
      k = cv2.waitKey(1)
      if k in (ord("q"), 27):
        raise KeyboardInterrupt
      if k == ord(" "):
        collecting = True
      p = st.track.pose
      if collecting and st.track.tracking_valid and p is not None and len(p.inlier_ids) >= min_faces:
        Rs.append(p.R)
        ts.append(st.depth.position_m if st.depth is not None and st.depth.position_m is not None else p.t)
        if len(Rs) >= frames_per_placement:
          break
    R, t, spread = average_pose(Rs, ts)
    caps.append({"name": ref["name"], "R_cam_cube": R.tolist(), "t_cam_cube": t.tolist(), "spread": spread})
  cv2.destroyAllWindows()
  return caps


# ----------------------------------------------------------------------------- inventory
def inventory(root: Path) -> dict:
  root = Path(root)
  vids = [p for ext in ("mp4", "mov", "avi", "mkv") for p in root.rglob(f"*.{ext}") if ".git" not in p.parts
          and ".venv" not in p.parts]
  logs = sorted((root / "LEAP_Hand_RL_Deploy_Kit" / "logs").glob("*.npz"))
  out = {"videos": [], "robot_logs": len(logs), "notes": []}
  import cv2

  for v in sorted(vids):
    cap = cv2.VideoCapture(str(v))
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    d = v.parent
    depth = [str(p) for p in list(d.glob("depth")) + list(d.glob("*depth*.npz")) if p.exists()]
    ts = (d / "frames.csv").is_file()
    out["videos"].append({"path": str(v.relative_to(root)), "frames": n, "fps": round(fps, 3), "size": [w, h],
                          "metric_depth": depth or None, "frame_timestamps": ts,
                          "synchronised_robot_log": None if not ts else "match frames.csv host times to a run log's "
                                                                       "started_at + elapsed_s"})
  out["notes"].append("A robot log can only be associated with a video that has absolute frame timestamps; otherwise a "
                      "manual --sync-offset-s is required and must be justified (e.g. a visible start event).")
  return out


# ----------------------------------------------------------------------------- synthetic demo
def make_synthetic_recording(out_dir: Path, seconds: float = 12.0, fps: float = 30.0, yaw_rate_deg_s: float = 90.0,
                             with_depth: bool = True, seed: int = 0) -> dict:
  """A spinning-cube recording with exact truth: rgb.mp4, frames.csv, depth/*.png (+meta), truth.jsonl,
  synthetic.intrinsics.json.  The yaw axis is the camera -Y axis (cube spinning about the vertical)."""
  import cv2

  from .geometry import load_cube_model, rotvec_to_matrix
  from .synthetic import CubeRenderer, degrade_depth, synthetic_intrinsics

  out = Path(out_dir)
  out.mkdir(parents=True, exist_ok=False)
  model = load_cube_model()
  intr = synthetic_intrinsics()
  intr.save(out / "synthetic.intrinsics.json")
  r = CubeRenderer(model, intr)
  rng = np.random.default_rng(seed)
  vw = cv2.VideoWriter(str(out / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, intr.image_size_wh)
  if with_depth:
    (out / "depth").mkdir()
    (out / "depth" / "depth_meta.json").write_text(json.dumps({"depth_scale_m_per_unit": 0.001, "aligned_to_color": True}))
  t0 = 1_800_000_000.0
  axis = np.array([0.0, -1.0, 0.0])
  tilt = rotvec_to_matrix(np.radians([-35.0, 0, 0]))  # camera looks down at the cube
  n = int(seconds * fps)
  with open(out / "frames.csv", "w") as fcsv, open(out / "truth.jsonl", "w") as ftruth:
    fcsv.write("frame,host_time_s,camera_time_s\n")
    for i in range(n):
      t = i / fps
      ang = np.radians(yaw_rate_deg_s) * t
      R = tilt @ rotvec_to_matrix(np.array([0, 0, 1.0]) * ang)  # spin about the cube's +Z (M face)
      p = np.array([0.01 * np.sin(0.5 * t), 0.0, 0.32])
      occl = [(300, 180, 340, 300)] if 4.0 <= t < 4.6 else None  # a "finger" passing in front
      drop = 7.0 <= t < 7.5  # 0.5 s with the cube hidden (tracking gap)
      fr = r.render(R, p, depth=with_depth, background_depth_m=0.6, occluders=occl, noise_std=1.5, rng=rng)
      img = fr.bgr
      if drop:
        img = np.full_like(img, 70)
      vw.write(img)
      fcsv.write(f"{i},{t0 + t:.6f},{t:.6f}\n")
      ftruth.write(json.dumps({"frame": i, "t": t, "R": R.tolist(), "p": p.tolist(), "spin_rad": float(ang),
                               "hidden": bool(drop)}) + "\n")
      if with_depth:
        dm = fr.depth_m if not drop else np.full_like(fr.depth_m, 0.6)
        cv2.imwrite(str(out / "depth" / f"depth_{i:06d}.png"), degrade_depth(dm, noise_std_m=0.0015, hole_fraction=0.05, rng=rng))
  vw.release()
  info = {"frames": n, "fps": fps, "yaw_rate_deg_s": yaw_rate_deg_s, "hidden_interval_s": [7.0, 7.5],
          "occlusion_interval_s": [4.0, 4.6], "true_total_spin_deg": yaw_rate_deg_s * (n - 1) / fps,
          "spin_axis": "cube +Z (M face normal), tilted 35 deg toward the camera", "created": _dt.datetime.now().isoformat()}
  (out / "synthetic_info.json").write_text(json.dumps(info, indent=2) + "\n")
  return info
