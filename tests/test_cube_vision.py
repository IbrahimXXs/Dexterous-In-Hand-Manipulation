"""Tests for training_source/project/real_world/cube_vision (read-only cube vision + dashboard).

Numerical pose tests use synthetic images rendered from the same layout files, with known
transforms; they check errors in metres and degrees, not mere detection.
Run:  uv run --extra vision pytest tests/test_cube_vision.py
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import socket
import sys
import types

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
cv2 = pytest.importorskip("cv2")
if not hasattr(cv2, "aruco"):
  pytest.skip("OpenCV without aruco (install the 'vision' extra)", allow_module_level=True)

REPO = Path(__file__).resolve().parents[1]
PROJ = REPO / "training_source" / "project"
sys.path.insert(0, str(PROJ))

from real_world.cube_vision import depth as D  # noqa: E402
from real_world.cube_vision import geometry as G  # noqa: E402
from real_world.cube_vision.extrinsics import (kabsch, load_camera_to_palm, save_camera_to_palm,  # noqa: E402
                                               solve_camera_to_palm)
from real_world.cube_vision.intrinsics import (STATUS_CHECKERBOARD, CameraIntrinsics, IntrinsicsError,  # noqa: E402
                                               load_intrinsics)
from real_world.cube_vision.synthetic import (CubeRenderer, degrade_depth, look_rotation, random_pose,  # noqa: E402
                                              synthetic_intrinsics)
from real_world.cube_vision.tracker import CubeTracker, MarkerDetector, TrackState  # noqa: E402
from real_world.cube_vision.yaw import YAW_AXIS_PALM, YawAccumulator  # noqa: E402

MODEL = G.load_cube_model()
INTR = synthetic_intrinsics()
REN = CubeRenderer(MODEL, INTR)


def track(img, tracker=None, t=0.0):
  return (tracker or CubeTracker(MODEL, INTR)).process(img, t)


# ----------------------------------------------------------------------------- 1-5 geometry
def test_layout_loads_six_faces_from_json():
  raw = json.loads(G.LAYOUT_PATH.read_text())
  assert len(MODEL.faces) == 6 and MODEL.marker_ids == (0, 1, 2, 3, 4, 5)
  for f in raw["faces"]:  # values come from the file, not duplicated constants
    face = MODEL.faces[f["id"]]
    assert face.name == f["name"] and face.letter == f["letter"]
    assert np.array_equal(face.marker_x, f["marker_x_in_cube"]) and np.array_equal(face.marker_y, f["marker_y_in_cube"])


def test_cube_and_marker_dimensions():
  assert MODEL.cube_size_m == pytest.approx(0.058) and MODEL.half_m == pytest.approx(0.029)
  assert MODEL.marker_size_m == pytest.approx(0.01555)
  assert MODEL.dictionary_name == "DICT_4X4_50" and MODEL.dictionary_id() == cv2.aruco.DICT_4X4_50
  assert MODEL.mass_kg == pytest.approx(0.085)


def test_face_axes_and_normals():
  for f in MODEL.faces.values():
    R = f.R_cube_marker
    assert np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(np.cross(f.marker_x, f.marker_y), f.normal)
    assert np.allclose(f.face_center, MODEL.half_m * f.normal)
  normals = {tuple(f.normal.astype(int)) for f in MODEL.faces.values()}
  assert len(normals) == 6


def test_marker_id_to_face_mapping():
  expected = {0: ("+Z", "M", "yellow"), 1: ("+X", "B", "red"), 2: ("+Y", "Z", "green"), 3: ("-X", "U", "blue"),
              4: ("-Y", "A", "white"), 5: ("-Z", "I", "orange")}
  assert {i: (f.name, f.letter, f.color) for i, f in MODEL.faces.items()} == expected


def test_marker_corner_positions_in_cube_frame():
  s, h = MODEL.marker_size_m, MODEL.half_m
  for f in MODEL.faces.values():
    c = f.corners
    assert np.allclose(c @ f.normal, h)                       # on the face plane
    assert np.allclose(c.mean(0), f.marker_center)
    for k in range(4):                                         # 15.55 mm square
      assert np.linalg.norm(c[k] - c[(k + 1) % 4]) == pytest.approx(s)
    assert np.allclose((c[1] - c[0]) / s, f.marker_x)          # TL -> TR along marker +x
    assert np.allclose((c[0] - c[3]) / s, f.marker_y)          # BL -> TL along marker +y (marker top)
    off = MODEL.print_layout.offset_for(f.marker_id)
    assert np.allclose(f.marker_center, f.face_center + off[0] * f.marker_x + off[1] * f.marker_y)
    assert np.all(np.abs(c - f.face_center) <= h + 1e-12)      # inside the face


def test_centered_hypothesis_and_bad_offset_rejected():
  m = G.load_cube_model(centered_markers=True)
  assert all(np.allclose(f.marker_center, f.face_center) for f in m.faces.values())
  with pytest.raises(G.LayoutError):
    G.load_cube_model(offset_override_m=(0.025, 0.0))  # marker would leave the face


def test_opencv_corner_order_matches_model_convention():
  """generateImageMarker's top-left is OpenCV's corner 0; the model maps corner 0 to (-s/2, +s/2)."""
  d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
  img = np.full((300, 300), 255, np.uint8)
  img[50:250, 50:250] = cv2.aruco.generateImageMarker(d, 3, 200, borderBits=1)
  det = MarkerDetector(MODEL)
  dets, *_ = det.detect(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
  assert [x.marker_id for x in dets] == [3]
  c = dets[0].corners
  assert np.allclose(c, [[49.5, 49.5], [249.5, 49.5], [249.5, 249.5], [49.5, 249.5]], atol=1.0)
  assert np.allclose(G.CORNERS_UNIT[0], [-0.5, 0.5]) and np.allclose(G.CORNERS_UNIT[2], [0.5, -0.5])


def test_detected_corners_follow_rotated_faces():
  """On every face (markers rotated differently on the cube), detected corner k is the projection of model corner k."""
  rng = np.random.default_rng(11)
  seen = set()
  for _ in range(40):
    R, t = random_pose(rng)
    fr = REN.render(R, t)
    tr = CubeTracker(MODEL, INTR)
    for d in tr.detect(fr.bgr)[0]:
      proj = tr._project(MODEL.faces[d.marker_id].corners, R, t)
      err = lambda P: float(np.mean(np.linalg.norm(d.corners - P, axis=1)))  # noqa: E731
      e0 = err(proj)
      others = [err(np.roll(proj, k, axis=0)) for k in (1, 2, 3)] + [err(proj[::-1])]
      assert e0 < 0.25 * d.side_px and e0 < 0.5 * min(others), (d.marker_id, e0, others)
      seen.add(d.marker_id)
  assert seen == set(range(6))


# ----------------------------------------------------------------------------- 6-9 pose
def test_multi_face_pose_recovery_metric_error():
  rng = np.random.default_rng(3)
  pe, re = [], []
  while len(pe) < 40:
    R, t = random_pose(rng)
    res = track(REN.render(R, t).bgr)
    if res.pose is None or len(res.pose.inlier_ids) < 2:
      continue
    assert res.state == TrackState.TRACKING and res.tracking_valid
    pe.append(np.linalg.norm(res.pose.t - t))
    re.append(G.rotation_angle_deg(res.pose.R, R))
  pe, re = np.array(pe), np.array(re)
  assert np.median(pe) < 0.003 and pe.max() < 0.015, pe      # metres
  assert np.median(re) < 1.5 and re.max() < 4.0, re           # degrees


def test_single_face_ambiguity_is_explicit_and_temporal_prior_resolves_it():
  # the -Z (I) face looks at the camera (its normal is -Z in the cube frame): only marker 5 is detectable
  R = G.rotvec_to_matrix(np.radians([6, 4, 0]))
  t = np.array([0.0, 0.0, 0.30])
  fr = REN.render(R, t)
  res = track(fr.bgr)
  assert res.visible_ids == (5,)
  assert res.pose is not None and len(res.pose.inlier_ids) == 1
  assert res.state == TrackState.PARTIAL
  if res.pose.ambiguous:
    assert not res.tracking_valid and res.pose.alt_rotation_deg > 1.0
  # with a recent prior near the truth the solution is chosen temporally and is accurate
  tr = CubeTracker(MODEL, INTR)
  tr.prev = (0.0, G.rotvec_to_matrix(np.radians([3, 2, 0])) @ R, t + 0.002)
  res2 = tr.process(fr.bgr, 0.05)
  assert res2.pose is not None and not res2.pose.ambiguous and res2.pose.disambiguation == "temporal"
  assert G.rotation_angle_deg(res2.pose.R, R) < 4.0 and np.linalg.norm(res2.pose.t - t) < 0.01


def test_partial_occlusion_keeps_tracking():
  rng = np.random.default_rng(21)
  for _ in range(200):
    R, t = random_pose(rng, (0.25, 0.30))
    full = track(REN.render(R, t).bgr)
    if full.pose is not None and len(full.pose.inlier_ids) == 3:
      break
  victim = full.detections[0]
  x0, y0 = np.floor(victim.corners.min(0)).astype(int) - 2
  x1, y1 = np.ceil(victim.corners.max(0)).astype(int) + 2
  res = track(REN.render(R, t, occluders=[(x0, y0, x1, y1)]).bgr)
  assert victim.marker_id not in res.visible_ids and len(res.visible_ids) == 2
  assert res.state == TrackState.TRACKING
  assert np.linalg.norm(res.pose.t - t) < 0.012 and G.rotation_angle_deg(res.pose.R, R) < 3.0


def test_unknown_ids_are_ignored_and_reported():
  R, t = G.rotvec_to_matrix(np.radians([30, 40, 10])), np.array([0.0, 0.0, 0.3])
  img = REN.render(R, t).bgr.copy()
  d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
  m = cv2.cvtColor(cv2.aruco.generateImageMarker(d, 17, 60, borderBits=1), cv2.COLOR_GRAY2BGR)
  img[20:100, 20:100] = 255
  img[30:90, 30:90] = m
  base = track(REN.render(R, t).bgr)
  res = track(img)
  assert 17 in res.unknown_ids and 17 not in res.visible_ids
  assert np.allclose(res.pose.t, base.pose.t, atol=1e-6)


def test_no_marker_reports_lost_not_a_pose():
  res = track(np.full((480, 640, 3), 90, np.uint8))
  assert res.state == TrackState.LOST and res.pose is None and not res.tracking_valid


# ----------------------------------------------------------------------------- 10 intrinsics
def test_invalid_or_missing_intrinsics(tmp_path):
  with pytest.raises(IntrinsicsError):
    load_intrinsics(tmp_path / "missing.json")
  bad = {"format": "leap_cube_vision_intrinsics/v1", "status": "uncalibrated", "camera_matrix": None}
  (tmp_path / "a.json").write_text(json.dumps(bad))
  with pytest.raises(IntrinsicsError):
    load_intrinsics(tmp_path / "a.json")
  good = INTR.to_json()
  good["camera_matrix"] = [[-1, 0, 320], [0, 600, 240], [0, 0, 1]]
  (tmp_path / "b.json").write_text(json.dumps(good))
  with pytest.raises(IntrinsicsError):
    load_intrinsics(tmp_path / "b.json")
  ok = INTR.to_json()
  ok["status"] = "uncalibrated"
  (tmp_path / "c.json").write_text(json.dumps(ok))
  with pytest.raises(IntrinsicsError):
    load_intrinsics(tmp_path / "c.json")
  with pytest.raises(IntrinsicsError):
    CubeTracker(MODEL, None)
  with pytest.raises(IntrinsicsError):  # never rescaled silently
    CubeTracker(MODEL, INTR).process(np.zeros((720, 1280, 3), np.uint8), 0.0)
  INTR.save(tmp_path / "d.json")
  assert load_intrinsics(tmp_path / "d.json").status == STATUS_CHECKERBOARD


# ----------------------------------------------------------------------------- 11-14 depth
def _view():
  return G.rotvec_to_matrix(np.radians([35, 30, 5])), np.array([0.01, -0.005, 0.33])


def test_rgb_only_pnp_metric_depth():
  R, t = _view()
  res = track(REN.render(R, t).bgr)
  dr = D.estimate_depth(res, None, INTR, MODEL)
  assert dr.source == D.DepthSource.PNP_METRIC and dr.depth_range_m is None
  assert abs(dr.z_m - t[2]) < 0.006 and dr.sigma_range_m > 0
  lost = track(np.full((480, 640, 3), 90, np.uint8))
  assert D.estimate_depth(lost, None, INTR, MODEL).source == D.DepthSource.DEPTH_UNAVAILABLE


def test_rgbd_alignment_scaling_and_fusion():
  R, t = _view()
  fr = REN.render(R, t, depth=True, background_depth_m=0.7)
  raw = degrade_depth(fr.depth_m, noise_std_m=0.001, scale_m_per_unit=0.001)
  assert raw.dtype == np.uint16
  dm = D.depth_to_meters(raw, 0.001)
  res = track(fr.bgr)
  dr = D.estimate_depth(res, dm, INTR, MODEL)
  assert dr.source == D.DepthSource.RGB_D_FUSED, dr.status
  assert abs(dr.depth_range_m - np.linalg.norm(t)) < 0.003
  assert abs(dr.range_m - np.linalg.norm(t)) < 0.003 and dr.sigma_range_m < dr.pnp_sigma_range_m
  assert dr.valid_ratio > 0.9 and dr.n_valid > 20
  # 0.1 mm units (scale 1e-4) give the same metres
  dm2 = D.depth_to_meters(degrade_depth(fr.depth_m, scale_m_per_unit=1e-4), 1e-4)
  assert np.nanmax(np.abs(dm2 - fr.depth_m)[fr.depth_m > 0]) < 1e-4
  # a depth image that is not aligned to the colour image is never fused
  dr2 = D.estimate_depth(res, dm[:-10], INTR, MODEL)
  assert dr2.source == D.DepthSource.PNP_METRIC and "aligned" in dr2.status
  with pytest.raises(ValueError):
    D.depth_to_meters(raw, None)


def test_invalid_and_zero_depth_not_fused():
  R, t = _view()
  fr = REN.render(R, t, depth=True, background_depth_m=0.7)
  res = track(fr.bgr)
  zero = D.depth_to_meters(np.zeros((480, 640), np.uint16), 0.001)
  assert np.isnan(zero).all()
  dr = D.estimate_depth(res, zero, INTR, MODEL)
  assert dr.source == D.DepthSource.PNP_METRIC and dr.valid_ratio == 0.0 and "rejected" in dr.status
  holes = D.depth_to_meters(degrade_depth(fr.depth_m, hole_fraction=0.8), 0.001)
  assert D.estimate_depth(res, holes, INTR, MODEL).source == D.DepthSource.PNP_METRIC
  far = np.full_like(fr.depth_m, 5.0)  # out of the valid sensor range
  assert D.estimate_depth(res, far, INTR, MODEL).source == D.DepthSource.PNP_METRIC


def test_pnp_vs_depth_consistency_checks():
  R, t = _view()
  fr = REN.render(R, t, depth=True, background_depth_m=0.7)
  res = track(fr.bgr)
  biased = D.estimate_depth(res, fr.depth_m + 0.03, INTR, MODEL)
  assert biased.source == D.DepthSource.PNP_METRIC and "range differs" in biased.status
  small = D.estimate_depth(res, fr.depth_m + 0.004, INTR, MODEL)
  assert small.source == D.DepthSource.RGB_D_FUSED
  lo, hi = sorted([small.pnp_range_m, small.depth_range_m])
  assert lo - 1e-9 <= small.range_m <= hi + 1e-9


# ----------------------------------------------------------------------------- 15 conventions + extrinsics
def test_quaternion_conventions_and_camera_to_palm(tmp_path):
  from real_world.bridge_aruco_pose import convert_tracker_record
  from real_world.cube_pose_io import rotation_matrix_from_quaternion_wxyz

  rng = np.random.default_rng(5)
  for _ in range(20):
    R, _ = random_pose(rng)
    q = G.quat_wxyz_from_matrix(R)
    assert q[0] >= 0 and np.allclose(rotation_matrix_from_quaternion_wxyz(q), R, atol=1e-9)
  # placeholder files are never used
  st = load_camera_to_palm(PROJ / "real_world" / "camera_to_palm_iphone.json")
  assert not st.calibrated and st.transform is None and "NOT calibrated" in st.message
  assert not load_camera_to_palm(None).calibrated
  # solver recovers a known transform from >= 2 placements
  R_pc = G.rotvec_to_matrix([0.3, -1.2, 0.4])
  t_pc = np.array([0.05, -0.12, 0.31])
  samples = []
  for k in range(3):
    R_cc, t_cc = random_pose(rng)
    samples.append({"R_cam_cube": R_cc, "t_cam_cube": t_cc, "R_palm_cube": R_pc @ R_cc, "t_palm_cube": R_pc @ t_cc + t_pc})
  sol = solve_camera_to_palm(samples, MODEL.cube_corners())
  assert np.allclose(sol["R_palm_camera"], R_pc, atol=1e-9) and np.allclose(sol["translation_palm_camera_m"], t_pc, atol=1e-9)
  save_camera_to_palm(tmp_path / "c2p.json", sol, reference="test", n_samples=3, accept=True)
  st = load_camera_to_palm(tmp_path / "c2p.json")
  assert st.calibrated
  # our palm transform == the existing bridge's conversion
  rec = {"tracking_valid": True, "cube_center_m": [0.01, 0.02, 0.3], "cube_rotation_matrix": samples[0]["R_cam_cube"].tolist(),
         "host_time_s": 1.0, "frame": 3}
  pk = convert_tracker_record(rec, st.transform.t, st.transform.R, np.array([1.0, 0, 0, 0]))
  assert np.allclose(pk["cube_position_palm_m"], st.transform.point(rec["cube_center_m"]), atol=1e-9)
  assert np.allclose(rotation_matrix_from_quaternion_wxyz(pk["cube_quaternion_palm_wxyz"]),
                     st.transform.rotation(samples[0]["R_cam_cube"]), atol=1e-6)
  save_camera_to_palm(tmp_path / "bad.json", sol, reference="test", n_samples=1, accept=False)
  assert not load_camera_to_palm(tmp_path / "bad.json").calibrated


def test_kabsch_handles_reflection_free_case():
  P = np.random.default_rng(0).normal(size=(10, 3))
  R = G.rotvec_to_matrix([0.1, 0.2, 0.3])
  R2, t2 = kabsch(P, P @ R.T + 1.0)
  assert np.allclose(R2, R) and np.allclose(t2, 1.0) and np.linalg.det(R2) > 0


# ----------------------------------------------------------------------------- 16 sync + telemetry
def test_timestamp_association_and_staleness():
  from real_world.cube_vision.telemetry import TelemetryBuffer, TelemetrySample

  b = TelemetryBuffer()
  for k in range(10):
    b.add(TelemetrySample(100.0 + 0.05 * k, {"step": k, "t": 0.05 * k}))
  s, age = b.nearest(100.12, 0.1)
  assert s.packet["step"] == 2 and age == pytest.approx(0.02)   # causal: latest at/before the frame
  s, age = b.nearest(101.0, 0.1)
  assert s is None and age == pytest.approx(0.55)                # stale -> no association
  assert b.nearest(99.0, 0.1) == (None, None)
  b.add(TelemetrySample(100.025, {"step": 99}))                  # out-of-order packets stay sorted
  assert b.nearest(100.03, 0.1)[0].packet["step"] == 99


def test_run_log_host_time_and_udp_read_only(tmp_path):
  from real_world.cube_vision.telemetry import UdpTelemetry

  u = UdpTelemetry("127.0.0.1:0")
  port = u.sock.getsockname()[1]
  tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  tx.sendto(json.dumps({"type": "step", "step": 7, "t": 1.0, "q_server": [0] * 16}).encode(), ("127.0.0.1", port))
  tx.sendto(b"not json", ("127.0.0.1", port))
  import time

  time.sleep(0.05)
  got = u.poll()
  assert [s.packet["step"] for s in got] == [7] and u.n_bad == 1
  u.close()
  with pytest.raises(ValueError):
    UdpTelemetry("127.0.0.1:47999", forward="127.0.0.1:47999")


def test_recording_timestamps_and_depth_files(tmp_path):
  from real_world.cube_vision.calib import make_synthetic_recording
  from real_world.cube_vision.sources import RecordingSource, SourceUnavailable

  make_synthetic_recording(tmp_path / "rec", seconds=0.5)
  src = RecordingSource(tmp_path / "rec" / "rgb.mp4", tmp_path / "rec" / "depth")
  f = src.read_at(5)
  assert f.host_time_s == pytest.approx(1_800_000_000.0 + 5 / 30.0) and f.depth_m is not None
  assert np.nanmedian(f.depth_m) > 0.2
  with pytest.raises(SourceUnavailable):
    RecordingSource(tmp_path / "rec" / "rgb.mp4", tmp_path / "rec" / "rgb.mp4")  # colourised depth video refused


# ----------------------------------------------------------------------------- 17 lost / recovered
def test_tracking_lost_and_recovered_events_and_records():
  from real_world.cube_vision.extrinsics import ExtrinsicsStatus
  from real_world.cube_vision.pipeline import VisionPipeline
  from real_world.cube_vision.records import frame_record
  from real_world.cube_vision.sources import Frame

  pipe = VisionPipeline(MODEL, INTR, ExtrinsicsStatus(False, "palm-frame pose NOT calibrated"))
  R, t = _view()
  good = REN.render(R, t).bgr
  blank = np.full_like(good, 70)
  recs = []
  for k in range(20):
    img = blank if 6 <= k < 12 else good
    st = pipe.process(Frame(k, k / 30, None, img))
    recs.append(frame_record(st))
  kinds = [e.message for e in pipe.events.events if e.kind == "tracking"]
  assert any("LOST" in m for m in kinds) and any("recovered" in m for m in kinds)
  lost = recs[8]
  assert lost["tracking_valid"] is False and "cube_center_m" not in lost and lost["tracking_state"] == "LOST"
  assert lost["palm"] is None and "NOT" in lost["palm_status"]


# ----------------------------------------------------------------------------- 18 JSONL compatibility
def test_jsonl_compatible_with_existing_bridge(tmp_path):
  from real_world.bridge_aruco_pose import convert_tracker_record, latest_jsonl_record
  from real_world.cube_vision.extrinsics import ExtrinsicsStatus
  from real_world.cube_vision.pipeline import VisionPipeline
  from real_world.cube_vision.records import JsonlWriter, frame_record
  from real_world.cube_vision.sources import Frame

  pipe = VisionPipeline(MODEL, INTR, ExtrinsicsStatus(False, "x"))
  R, t = _view()
  w = JsonlWriter(tmp_path / "poses.jsonl")
  for k in range(3):
    w.write(frame_record(pipe.process(Frame(k, 10 + k / 30, None, REN.render(R, t).bgr))))
  w.close()
  rec = latest_jsonl_record(tmp_path / "poses.jsonl")
  for key in ("frame", "host_time_s", "tracking_valid", "cube_center_m", "cube_rotation_matrix", "visible_marker_ids",
              "reprojection_error_px", "depth_source", "cube_depth_z_m", "depth_valid_ratio", "n_visible_markers",
              "inlier_corner_count", "position_std_m", "pose_confidence", "camera_time_s"):
    assert key in rec, key
  pk = convert_tracker_record(rec, np.zeros(3), np.eye(3), np.array([1.0, 0, 0, 0]))
  assert np.allclose(pk["cube_position_palm_m"], rec["cube_center_m"]) and pk["source_frame"] == 2
  bad = dict(rec, tracking_valid=False)
  with pytest.raises(ValueError):
    convert_tracker_record(bad, np.zeros(3), np.eye(3), np.array([1.0, 0, 0, 0]))


# ----------------------------------------------------------------------------- 19 yaw
def _spin(n_rev, n=600, axis=YAW_AXIS_PALM, noise_deg=0.3, sign_flip=True, seed=0):
  rng = np.random.default_rng(seed)
  base = G.rotvec_to_matrix([0.2, -0.1, 0.3])
  out = []
  for k in range(n):
    ang = 2 * np.pi * n_rev * k / (n - 1)
    R = G.rotvec_to_matrix(axis * ang) @ base @ G.rotvec_to_matrix(np.radians(rng.normal(0, noise_deg, 3)))
    q = G.quat_wxyz_from_matrix(R)
    if sign_flip and k % 2:
      q = -q  # quaternion sign must not matter
    out.append((k / 30.0, G.matrix_from_quat_wxyz(q)))
  return out


def test_yaw_accumulates_multiple_revolutions_signed():
  for n_rev in (3.5, -2.25):
    acc = YawAccumulator(YAW_AXIS_PALM.copy(), "palm")
    for t, R in _spin(n_rev):
      acc.update(t, R)
    s = acc.summary()
    assert s["yaw_total_rad"] == pytest.approx(2 * np.pi * n_rev, abs=np.radians(2))
    assert s["revolutions"] == pytest.approx(abs(n_rev), abs=0.01)
    assert s["quarter_turns"] == pytest.approx(4 * abs(n_rev), abs=0.04)  # one revolution = four quarter turns
    assert s["direction"] == ("+" if n_rev > 0 else "-")


def test_yaw_axis_estimated_without_calibration_and_gaps():
  acc = YawAccumulator(None, "camera")
  ax = np.array([0.3, -0.8, 0.5])
  ax /= np.linalg.norm(ax)
  seq = _spin(2.0, axis=ax)
  for k, (t, R) in enumerate(seq):
    if 200 <= k < 205:      # short gap (0.17 s): bridged
      continue
    if 400 <= k < 460:      # long gap (2 s, > 1 rev/s): excluded, not invented
      continue
    acc.update(t, R)
  s = acc.summary()
  assert "estimated from motion" in s["axis_source"]
  assert s["bridged_intervals"] == 1 and s["excluded_intervals"] == 1
  lost = 2 * np.pi * 2.0 * 60 / 599
  assert abs(s["yaw_total_rad"]) == pytest.approx(4 * np.pi - lost, abs=np.radians(3))
  assert s["excluded_time_s"] == pytest.approx(61 / 30, abs=0.05)


def test_yaw_rejects_implausible_jump():
  acc = YawAccumulator(YAW_AXIS_PALM.copy(), "palm")
  acc.update(0.0, np.eye(3))
  acc.update(1 / 30, G.rotvec_to_matrix(YAW_AXIS_PALM * np.radians(170)))
  assert acc.yaw_rad == 0.0 and acc.intervals[-1].kind == "rejected_jump"


# ----------------------------------------------------------------------------- 20 dashboard replay (no hardware)
def test_dashboard_replay_synthetic_no_hardware(tmp_path):
  from real_world.cube_vision import app
  from real_world.cube_vision.calib import make_synthetic_recording
  from real_world.cube_vision.dashboard import Dashboard, ReplayIndex, ViewInfo

  info = make_synthetic_recording(tmp_path / "rec", seconds=2.0)
  args = types.SimpleNamespace(video=str(tmp_path / "rec" / "rgb.mp4"), depth=str(tmp_path / "rec" / "depth"),
                               timestamps=None, depth_scale=None, intrinsics=str(tmp_path / "rec" / "synthetic.intrinsics.json"),
                               layout=None, print_layout=None, camera_to_palm=None, log=None, sync_offset_s=0.0,
                               out=str(tmp_path / "out"), max_frames=None, camera_yaw_axis=None)
  a = app.analyse_recording(args, progress=False)
  states = a["states"]
  assert len(states) == info["frames"]
  lines = (tmp_path / "out" / "poses.jsonl").read_text().splitlines()
  assert len(lines) == info["frames"]
  for name in ("poses.csv", "annotated.mp4", "summary.json", "summary.md"):
    assert (tmp_path / "out" / name).stat().st_size > 0
  s = a["summary"]
  assert s["state_fraction"].get("TRACKING", 0) > 0.9 and s["depth_source_fraction"].get("RGB_D_FUSED", 0) > 0.85
  true = np.radians(info["yaw_rate_deg_s"]) * (info["frames"] - 1) / info["fps"]
  assert abs(s["yaw"]["yaw_total_rad"]) == pytest.approx(true, abs=np.radians(4))
  dash = Dashboard(a["model"], a["intr"], with_twin=False)
  img = dash.compose(states[30], a["src"].read_at(30).bgr, a["src"].read_at(30).depth_m, ReplayIndex(states).history(30),
                     a["buf"], a["pipe"].events.events, ViewInfo("REPLAY", "test", True, "none", "none", n_frames=len(states)),
                     False)
  assert img.shape == (1000, 1680, 3) and img.std() > 10


def test_twin_renders_offscreen_when_gl_available():
  from real_world.cube_vision.twin import TwinRenderer

  try:
    tw = TwinRenderer(MODEL, 160, 120)
  except Exception as exc:  # noqa: BLE001
    pytest.skip(f"no offscreen GL here: {exc}")
  img = tw.render(None, None, (np.array([0.0, 0.0, 0.05]), np.eye(3)))
  assert img.shape == (120, 160, 3) and img.std() > 5


# ----------------------------------------------------------------------------- layout check + read-only guarantee
def test_layout_check_votes_identify_layout_on_synthetic_data():
  from real_world.cube_vision import layout_check as L

  rng = np.random.default_rng(9)
  frames = []
  det = MarkerDetector(MODEL)
  while len(frames) < 60:
    R, t = random_pose(rng, (0.22, 0.32), 0.02)
    d = det.detect(REN.render(R, t).bgr)[0]
    if len(d) >= 2:
      frames.append(L.FrameObs(len(frames), 0.0, d))
  pairs = L.relative_rotation_votes(frames, MODEL, (640, 480))
  assert pairs and all(p["matches_layout_json"] for p in pairs.values())
  rec = L.reconstruct_marker_frames(pairs, MODEL)
  assert all(v["axes_match"] for v in rec["per_marker"].values() if v["observed"])


def test_measured_candidate_layout_files_load_and_are_labelled():
  mdir = PROJ / "assets" / "mbzuai_cube" / "measured"
  lay = json.loads((mdir / "layout.measured_rgb.json").read_text())
  assert lay["status"] == "measured_from_recording_unconfirmed"
  m = G.load_cube_model(mdir / "layout.measured_rgb.json", mdir / "marker_print_layout.measured_rgb.json")
  assert m.print_layout.status == "consistent_with_recording"
  assert json.loads(G.LAYOUT_PATH.read_text())["faces"][3]["name"] == "-X"  # layout.json itself unchanged


def test_package_is_read_only():
  """No motor / serial / RPC access anywhere in the vision package."""
  pkg = PROJ / "real_world" / "cube_vision"
  src = "\n".join(p.read_text() for p in pkg.glob("*.py")) + (REPO / "scripts" / "cube_vision.py").read_text()
  for forbidden in ("dynamixel", "import serial", "from serial", "zmq", "enable_torque", "torque_enable", "set_goal",
                    "write_goal", "import deploy_direct", "kit_import(\"deploy_direct\")", "subprocess"):
    assert forbidden not in src, forbidden
  # the only UDP send is the optional telemetry relay to a *different* local viewer port
  assert src.count(".sendto(") == 1 and "self.fwd[1].sendto(data, self.fwd[0])" in src


def test_live_loop_with_stream_source_and_udp_telemetry(tmp_path):
  """The live code path end to end without hardware: an OpenCV 'stream' (a file) + kit-format UDP packets."""
  import threading
  import time

  from real_world.cube_vision import app
  from real_world.cube_vision.calib import make_synthetic_recording

  make_synthetic_recording(tmp_path / "rec", seconds=1.0, with_depth=False)
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  s.bind(("127.0.0.1", 0))
  port = s.getsockname()[1]
  s.close()
  stop = threading.Event()

  def sender():
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    k = 0
    while not stop.is_set():
      pk = {"type": "step", "step": k, "t": 0.05 * k, "shadow": True, "port": "SIMULATED", "policy": "test",
            "q_server": [0.1] * 16, "cmd_server": [0.12] * 16, "current_ma_hw": [100.0] * 15 + [400.0], "temp_max_c": 30.0,
            "monitor": {"cube_pos_palm": [0, 0, 0.05], "cube_quat_palm": [1, 0, 0, 0], "est_yaw_rad": 0.01 * k}}
      tx.sendto(json.dumps(pk).encode(), ("127.0.0.1", port))
      k += 1
      time.sleep(0.05)

  th = threading.Thread(target=sender, daemon=True)
  th.start()
  time.sleep(0.2)
  args = types.SimpleNamespace(camera=str(tmp_path / "rec" / "rgb.mp4"), realsense_serial=None, width=None, height=None,
                               fps=None, intrinsics=str(tmp_path / "rec" / "synthetic.intrinsics.json"), layout=None,
                               print_layout=None, camera_to_palm=None, udp=f"127.0.0.1:{port}", forward_udp=None,
                               out=str(tmp_path / "live"), jsonl=None, record=True, headless=True, no_twin=True,
                               max_seconds=20, camera_yaw_axis=None)
  try:
    assert app.run_live(args) == 0
  finally:
    stop.set()
  summ = json.loads((tmp_path / "live" / "summary.json").read_text())
  assert summ["mode"] == "LIVE" and summ["frames"] == 30
  assert summ["event_counts"].get("telemetry", 0) >= 1          # telemetry received
  assert any("high motor current th_ipl" in e["message"] for e in summ["events"])  # 400 mA >= 0.9 x 350 mA
  assert (tmp_path / "live" / "recording" / "rgb.mp4").stat().st_size > 0
  rec = [json.loads(x) for x in (tmp_path / "live" / "poses.jsonl").read_text().splitlines()]
  assert any("telemetry" in r for r in rec)


def test_vision_vs_proprio_comparison_needs_calibration_and_is_symmetry_aware(tmp_path):
  from real_world.cube_vision.extrinsics import ExtrinsicsStatus
  from real_world.cube_vision.pipeline import VisionPipeline
  from real_world.cube_vision.sources import Frame
  from real_world.cube_vision.telemetry import TelemetryBuffer, TelemetrySample

  # a measured-looking transform (not identity): camera above the palm
  R_pc = G.rotvec_to_matrix([2.6, 0.2, -0.3])
  t_pc = np.array([0.02, -0.03, 0.31])
  sol = {"R_palm_camera": R_pc, "translation_palm_camera_m": t_pc, "residuals": [], "max_center_error_mm": 0.0,
         "max_rotation_error_deg": 0.0}
  save_camera_to_palm(tmp_path / "c2p.json", sol, reference="test", n_samples=3, accept=True)
  extr = load_camera_to_palm(tmp_path / "c2p.json")
  sym = G.cube_symmetry_rotations()[7]
  for calibrated in (False, True):
    buf = TelemetryBuffer()
    pipe = VisionPipeline(MODEL, INTR, extr if calibrated else ExtrinsicsStatus(False, "NOT calibrated"), telemetry=buf)
    R0, t = _view()
    spin_axis_cam = R_pc.T @ YAW_AXIS_PALM   # spin about the palm yaw axis, seen from the camera
    out = []
    for k in range(40):
      ang = np.radians(3.0) * k
      R = G.rotvec_to_matrix(spin_axis_cam * ang) @ R0
      host = 100.0 + k / 30.0
      R_palm, p_palm = R_pc @ R, R_pc @ t + t_pc
      mon = {"cube_pos_palm": p_palm.tolist(), "cube_quat_palm": G.quat_wxyz_from_matrix(R_palm @ sym).tolist(),
             "est_yaw_rad": float(ang)}
      buf.add(TelemetrySample(host - 0.01, {"step": k, "t": k / 30, "q_server": [0] * 16, "monitor": mon}))
      out.append(pipe.process(Frame(k, host, None, REN.render(R, t).bgr)))
    if not calibrated:
      assert all(s.comparison is None for s in out) and all(s.palm is None for s in out)
      continue
    c = out[-1].comparison
    assert c is not None
    assert c["position_disagreement_mm"] < 8.0
    assert c["orientation_disagreement_deg_mod_symmetry"] < 3.0
    assert abs(c["cumulative_yaw_difference_deg"]) < 3.0
    assert pipe.yaw.frame == "palm" and pipe.yaw.yaw_rad == pytest.approx(np.radians(3.0 * 39), abs=np.radians(2.5))
