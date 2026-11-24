#!/usr/bin/env python3
"""MBZUAI cube vision + digital-twin dashboard (read-only; never commands the hand).

Install once (repo root):   uv sync --extra vision            (+ --extra realsense for Intel RealSense)

  replay              replay a recording (video [+ depth] [+ run log]) in the dashboard / export
  live                live camera (+ the kit's UDP telemetry) dashboard
  track               headless tracking of a recording -> poses.jsonl / poses.csv / annotated.mp4 / summary
  inventory           list recordings and which data sources each has
  list-cameras        probe OpenCV cameras and RealSense devices
  make-board          write a printable ChArUco board (+ its spec)
  calibrate-intrinsics  board calibration -> *.intrinsics.json
  verify-device-intrinsics  check factory intrinsics against a board calibration
  camera-to-palm      capture (live) / solve the camera-to-palm transform from known placements
  verify-layout       check the physical marker layout against layout.json from a recording
  synth               write a synthetic test recording with exact truth

See docs/live_vision_digital_twin.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training_source" / "project"))


def _common(p):
  p.add_argument("--layout", help="cube layout (default: assets/mbzuai_cube/layout.json)")
  p.add_argument("--print-layout", help="marker print layout (default: assets/mbzuai_cube/marker_print_layout.json)")
  p.add_argument("--intrinsics", help="camera intrinsics JSON (required except RealSense live)")
  p.add_argument("--camera-to-palm", help="camera-to-palm JSON (must be calibrated=true to be used)")
  p.add_argument("--camera-yaw-axis", nargs=3, type=float, help="camera-frame yaw axis when camera->palm is not calibrated "
                 "(default: estimated from the motion, sign arbitrary)")
  p.add_argument("--no-twin", action="store_true", help="do not render the 3D MuJoCo twin")


def _replay_args(p):
  _common(p)
  p.add_argument("--video", required=True)
  p.add_argument("--depth", help="metric depth: directory of 16-bit PNGs (+depth_meta.json) or .npz with 'depth'")
  p.add_argument("--depth-scale", type=float, help="metres per depth unit (if not in the depth files)")
  p.add_argument("--timestamps", help="frames.csv with frame,host_time_s,camera_time_s (default: next to the video)")
  p.add_argument("--log", help="kit run log (.npz) to synchronise")
  p.add_argument("--sync-offset-s", type=float, default=0.0,
                 help="video without absolute timestamps: run time = video time + offset; with timestamps: extra offset")
  p.add_argument("--out", help="output directory (poses.jsonl, poses.csv, annotated.mp4, summary.json/.md)")
  p.add_argument("--max-frames", type=int)


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = ap.add_subparsers(dest="cmd", required=True)

  p = sub.add_parser("replay")
  _replay_args(p)
  p.add_argument("--headless", action="store_true", help="no window: analyse + export only")
  p.add_argument("--screenshot-at", type=float, help="write a dashboard PNG at this experiment time [s]")
  p.add_argument("--dashboard-video", action="store_true", help="also export the full dashboard as dashboard.mp4")
  p.add_argument("--dashboard-video-stride", type=int, default=1)

  p = sub.add_parser("track")
  _replay_args(p)

  p = sub.add_parser("live")
  _common(p)
  p.add_argument("--camera", required=True, help="'realsense', an OpenCV index (0, 1, ...) or a stream URL")
  p.add_argument("--realsense-serial")
  p.add_argument("--width", type=int)
  p.add_argument("--height", type=int)
  p.add_argument("--fps", type=float)
  p.add_argument("--udp", help="listen for the kit's telemetry, e.g. 127.0.0.1:47800 (start the run with "
                 "./leap.sh run <policy> -- --live-udp 127.0.0.1:47800)")
  p.add_argument("--forward-udp", help="relay telemetry datagrams to a kit live_viewer on another port")
  p.add_argument("--out", help="output directory (default outputs/cube_vision/live_<time>)")
  p.add_argument("--jsonl", help="also/instead append poses to this JSONL (input of bridge_aruco_pose.py)")
  p.add_argument("--record", action="store_true", help="record rgb.mp4 + frames.csv (+ depth PNGs) for replay")
  p.add_argument("--headless", action="store_true")
  p.add_argument("--max-seconds", type=float)

  sub.add_parser("inventory")
  sub.add_parser("list-cameras")

  p = sub.add_parser("make-board")
  p.add_argument("--out", default="charuco_board.png")
  p.add_argument("--dpi", type=int, default=300)

  p = sub.add_parser("calibrate-intrinsics")
  p.add_argument("--out", required=True, help="output *.intrinsics.json")
  p.add_argument("--camera", help="live: 'realsense' or OpenCV index/URL")
  p.add_argument("--width", type=int)
  p.add_argument("--height", type=int)
  p.add_argument("--images", help="glob of board images")
  p.add_argument("--video", help="video of the board (every --every frames)")
  p.add_argument("--every", type=int, default=15)
  p.add_argument("--views", type=int, default=25)
  p.add_argument("--board", choices=("charuco", "chessboard"), default="charuco")
  p.add_argument("--squares-x", type=int, default=7)
  p.add_argument("--squares-y", type=int, default=5)
  p.add_argument("--square-m", type=float, default=0.030)
  p.add_argument("--marker-m", type=float, default=0.022)
  p.add_argument("--max-rms-px", type=float, default=1.0)
  p.add_argument("--min-views", type=int, default=12)
  p.add_argument("--camera-id", default=None)

  p = sub.add_parser("verify-device-intrinsics")
  p.add_argument("--device", required=True, help="device intrinsics JSON (e.g. saved from RealSense)")
  p.add_argument("--board-calibration", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--max-px", type=float, default=1.0)

  p = sub.add_parser("camera-to-palm")
  _common(p)
  p.add_argument("action", choices=("capture", "solve"))
  p.add_argument("--reference", required=True, help="known palm-frame cube placements (see camera_to_palm_reference.example.json)")
  p.add_argument("--captures", required=True, help="captures JSON (written by capture, read by solve)")
  p.add_argument("--camera")
  p.add_argument("--realsense-serial")
  p.add_argument("--width", type=int)
  p.add_argument("--height", type=int)
  p.add_argument("--fps", type=float)
  p.add_argument("--frames-per-placement", type=int, default=60)
  p.add_argument("--out", help="solve: output camera_to_palm JSON")
  p.add_argument("--max-center-mm", type=float, default=3.0)
  p.add_argument("--max-rotation-deg", type=float, default=2.0)

  p = sub.add_parser("verify-layout")
  p.add_argument("--video", required=True)
  p.add_argument("--out-dir", required=True)
  p.add_argument("--no-candidate", action="store_true")

  p = sub.add_parser("synth")
  p.add_argument("--out", required=True)
  p.add_argument("--seconds", type=float, default=12.0)
  p.add_argument("--no-depth", action="store_true")

  a = ap.parse_args()
  if a.cmd in ("replay", "track"):
    from real_world.cube_vision import app

    if a.cmd == "track":
      a.headless, a.screenshot_at, a.dashboard_video, a.dashboard_video_stride, a.no_twin = True, None, False, 1, True
      if not a.out:
        ap.error("track needs --out")
    return app.run_replay(a)
  if a.cmd == "live":
    from real_world.cube_vision import app

    return app.run_live(a)
  if a.cmd == "inventory":
    from real_world.cube_vision.calib import inventory

    print(json.dumps(inventory(REPO), indent=2))
    return 0
  if a.cmd == "list-cameras":
    from real_world.cube_vision.sources import list_cameras

    print(json.dumps(list_cameras(), indent=2))
    return 0
  if a.cmd == "make-board":
    from real_world.cube_vision.calib import make_board

    p = make_board(Path(a.out), dpi=a.dpi)
    print(f"wrote {p} and {p.with_suffix('.json')} - print at 100 % and measure a square")
    return 0
  if a.cmd == "calibrate-intrinsics":
    return _calibrate(a)
  if a.cmd == "verify-device-intrinsics":
    from real_world.cube_vision.calib import verify_device_intrinsics
    from real_world.cube_vision.intrinsics import load_intrinsics

    out, cmp = verify_device_intrinsics(load_intrinsics(Path(a.device)), load_intrinsics(Path(a.board_calibration)), a.max_px)
    out.save(Path(a.out))
    print(json.dumps({"status": out.status, **cmp}, indent=2))
    return 0
  if a.cmd == "camera-to-palm":
    return _camera_to_palm(a)
  if a.cmd == "verify-layout":
    from real_world.cube_vision import layout_check

    rep = layout_check.run(Path(a.video), Path(a.out_dir), write_candidate=not a.no_candidate)
    for k, v in rep["relative_rotation_pairs"].items():
      print(f"pair {k}: {v['frames']} frames, matches layout.json: {v['matches_layout_json']}")
    for g, res in rep["bundle_adjustment"].items():
      for n, r in res.items():
        print(f"{g:18s} {n:26s} rms {r['rms_px']:.2f} px")
    return 0
  if a.cmd == "synth":
    from real_world.cube_vision.calib import make_synthetic_recording

    print(json.dumps(make_synthetic_recording(Path(a.out), a.seconds, with_depth=not a.no_depth), indent=2))
    return 0
  return 1


def _calibrate(a) -> int:
  from real_world.cube_vision.calib import DEFAULT_BOARD, calibrate_intrinsics, capture_board_views, collect_images

  if a.board == "charuco":
    spec = dict(DEFAULT_BOARD, squares_x=a.squares_x, squares_y=a.squares_y, square_m=a.square_m, marker_m=a.marker_m)
  else:
    spec = {"type": "chessboard", "cols": a.squares_x - 1, "rows": a.squares_y - 1, "square_m": a.square_m}
  cam_id = a.camera_id
  if a.camera:
    from real_world.cube_vision.sources import OpenCVCameraSource, RealSenseSource

    src = RealSenseSource(None, a.width or 848, a.height or 480) if a.camera == "realsense" else \
      OpenCVCameraSource(a.camera, a.width, a.height)
    cam_id = cam_id or src.camera_id
    images = capture_board_views(src, spec, a.views)
    src.close()
  else:
    images = collect_images(a.images, a.video, a.every)
    cam_id = cam_id or "unknown (set --camera-id)"
  intr, rep = calibrate_intrinsics(images, spec, cam_id, max_rms_px=a.max_rms_px, min_views=a.min_views)
  intr.save(Path(a.out))
  print(json.dumps({k: v for k, v in rep.items() if k != "per_view_rms_px"}, indent=2))
  print(f"wrote {a.out} with status {intr.status}")
  return 0 if rep["accepted"] else 2


def _camera_to_palm(a) -> int:
  import numpy as np

  from real_world.cube_vision.extrinsics import load_reference_poses, save_camera_to_palm, solve_camera_to_palm

  ref_doc = json.loads(Path(a.reference).read_text())
  if ref_doc.get("is_example", False):
    raise SystemExit("ERROR: the reference file is the EXAMPLE template; enter the physically measured jig poses first")
  ref = load_reference_poses(Path(a.reference))
  if a.action == "capture":
    from real_world.cube_vision import app
    from real_world.cube_vision.calib import capture_reference_poses
    from real_world.cube_vision.extrinsics import ExtrinsicsStatus
    from real_world.cube_vision.pipeline import VisionPipeline

    if not a.camera:
      raise SystemExit("capture needs --camera")
    model = app.build_model(a)
    src = app.open_camera(a)
    intr = app.resolve_intrinsics(a, src.device_intrinsics())
    pipe = VisionPipeline(model, intr, ExtrinsicsStatus(False, "calibrating"))
    caps = capture_reference_poses(src, pipe, ref, a.frames_per_placement)
    src.close()
    Path(a.captures).write_text(json.dumps({"intrinsics": intr.source_path or intr.camera_id, "layout": str(model.layout_path),
                                            "captures": caps}, indent=2) + "\n")
    print(f"wrote {a.captures}")
    return 0
  caps = {c["name"]: c for c in json.loads(Path(a.captures).read_text())["captures"]}
  samples = []
  for r in ref:
    if r["name"] not in caps:
      raise SystemExit(f"no capture for placement {r['name']}")
    c = caps[r["name"]]
    samples.append({"R_cam_cube": np.asarray(c["R_cam_cube"]), "t_cam_cube": np.asarray(c["t_cam_cube"]),
                    "R_palm_cube": r["R_palm_cube"], "t_palm_cube": r["t_palm_cube"]})
  from real_world.cube_vision.geometry import load_cube_model

  sol = solve_camera_to_palm(samples, load_cube_model().cube_corners())
  ok = sol["max_center_error_mm"] <= a.max_center_mm and sol["max_rotation_error_deg"] <= a.max_rotation_deg and len(samples) >= 2
  if not a.out:
    raise SystemExit("solve needs --out")
  save_camera_to_palm(Path(a.out), sol, reference=ref_doc.get("reference", a.reference), n_samples=len(samples), accept=ok,
                      thresholds={"max_center_mm": a.max_center_mm, "max_rotation_deg": a.max_rotation_deg, "min_placements": 2})
  print(json.dumps({"calibrated": ok, "max_center_error_mm": sol["max_center_error_mm"],
                    "max_rotation_error_deg": sol["max_rotation_error_deg"], "residuals": sol["residuals"]}, indent=2))
  return 0 if ok else 2


if __name__ == "__main__":
  raise SystemExit(main())
