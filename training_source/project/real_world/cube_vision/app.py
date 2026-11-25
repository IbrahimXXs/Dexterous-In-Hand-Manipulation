"""Replay / live / export runners behind scripts/cube_vision.py."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np

from .dashboard import Dashboard, ReplayIndex, ViewInfo, annotate_camera, depth_view
from .extrinsics import load_camera_to_palm
from .geometry import LAYOUT_PATH, PRINT_LAYOUT_PATH, load_cube_model
from .intrinsics import IntrinsicsError, load_intrinsics
from .pipeline import PipelineConfig, VisionPipeline
from .records import CsvWriter, JsonlWriter, frame_record, write_summary
from .sources import Frame, OpenCVCameraSource, RealSenseSource, Recorder, RecordingSource, SourceUnavailable
from .telemetry import TelemetryBuffer, UdpTelemetry, load_run_log

WINDOW = "LEAP cube vision - digital twin (read-only)"


def build_model(args):
  return load_cube_model(Path(args.layout) if args.layout else LAYOUT_PATH,
                         Path(args.print_layout) if args.print_layout else PRINT_LAYOUT_PATH)


def resolve_intrinsics(args, device=None):
  if getattr(args, "intrinsics", None):
    return load_intrinsics(Path(args.intrinsics))
  if device is not None:
    return device
  raise SystemExit("ERROR: no camera intrinsics. Pass --intrinsics FILE (calibrate with: scripts/cube_vision.py "
                   "calibrate-intrinsics ...). There is no default camera matrix.")


def _axis(args):
  return None if not getattr(args, "camera_yaw_axis", None) else tuple(float(v) for v in args.camera_yaw_axis)


def _guards(args) -> dict:
  roi = getattr(args, "roi", None)
  return {"roi_xyxy": None if not roi else tuple(int(v) for v in str(roi).split(",")),
          "max_range_m": getattr(args, "max_range_m", None)}


# ----------------------------------------------------------------------------- replay
def analyse_recording(args, *, progress: bool = True):
  model = build_model(args)
  src = RecordingSource(Path(args.video), Path(args.depth) if args.depth else None,
                        Path(args.timestamps) if args.timestamps else None, args.depth_scale)
  intr = resolve_intrinsics(args)
  intr.check_frame(*src.size)
  extr = load_camera_to_palm(Path(args.camera_to_palm) if args.camera_to_palm else None)
  buf = TelemetryBuffer()
  runlog = None
  if args.log:
    runlog = load_run_log(Path(args.log))
    for s in runlog.samples:
      buf.add(s)
  offset = float(args.sync_offset_s or 0.0)
  if src.timestamps is not None:
    time_mode = "absolute"
    sync = f"sync: absolute frame timestamps" + (f" + offset {offset:+.3f} s" if offset else "")
  elif runlog is not None:
    time_mode = "manual"
    sync = (f"sync: MANUAL offset - run time = video time {offset:+.3f} s (video has no absolute timestamps)")
  else:
    time_mode = "video"
    sync = "no telemetry: camera-only replay"
  cfg = PipelineConfig(telemetry_offset_s=offset if time_mode == "absolute" else 0.0, camera_yaw_axis=_axis(args),
                       **_guards(args))
  pipe = VisionPipeline(model, intr, extr, cfg=cfg, telemetry=buf)
  out = Path(args.out) if getattr(args, "out", None) else None
  writers = []
  vw = None
  if out is not None:
    out.mkdir(parents=True, exist_ok=True)
    for p in (out / "poses.jsonl",):
      if p.exists():
        p.unlink()
    writers = [JsonlWriter(out / "poses.jsonl"), CsvWriter(out / "poses.csv")]
    if getattr(args, "annotated_video", True):
      import cv2

      vw = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), src.fps, src.size)
  states = []
  t0 = time.time()
  stop = src.n if not getattr(args, "max_frames", None) else min(src.n, args.max_frames)
  for i in range(stop):
    f = src.read_at(i)
    if f is None:
      break
    vt = i / src.fps
    if time_mode == "manual":
      f.host_time_s = runlog.started_at.timestamp() + vt + offset
    elif time_mode == "video":
      f.host_time_s = vt
    st = pipe.process(f, vt)
    if len(buf.times):
      pipe.events.telemetry(st.t_exp, st.telemetry is not None, "(no packet near this frame)")
    states.append(st)
    if writers:
      rec = frame_record(st)
      for w in writers:
        w.write(rec)
    if vw is not None:
      vw.write(annotate_camera(f.bgr, st, model, intr))
    if progress and i % 200 == 0:
      print(f"[cube-vision] analysed {i}/{stop} frames ({time.time() - t0:.1f} s)", flush=True)
  if runlog is not None and states:
    # the run's stop reason, placed at the last logged step (in experiment time)
    t_end = runlog.samples[-1].host_time_s - (states[0].track.host_time_s - states[0].t_exp)
    pipe.events.guard(t_end, runlog.stop_reason)
  for w in writers:
    w.close()
  if vw is not None:
    vw.release()
  summary = pipe.summary("REPLAY", str(args.video))
  summary.update({"sync": sync, "telemetry_log": str(args.log) if args.log else None,
                  "run_stop_reason": runlog.stop_reason if runlog else None,
                  "telemetry_frames_associated": int(sum(s.telemetry is not None for s in states)),
                  "depth_input": str(args.depth) if args.depth else None, "frames_analysed": len(states),
                  "print_layout_status": model.print_layout.status, "layout_file": str(model.layout_path)})
  if out is not None:
    write_summary(out / "summary.json", summary)
  return dict(model=model, src=src, intr=intr, extr=extr, buf=buf, runlog=runlog, pipe=pipe, states=states,
              sync=sync, summary=summary)


def run_replay(args) -> int:
  a = analyse_recording(args)
  src, states = a["src"], a["states"]
  if not states:
    print("no frames")
    return 1
  dash = Dashboard(a["model"], a["intr"], with_twin=not args.no_twin)
  idx = ReplayIndex(states)
  info = ViewInfo("REPLAY", src.label, src.has_depth,
                  ("telemetry: " + Path(args.log).name + f" ({a['runlog'].monitor_source})") if a["runlog"] else "telemetry: none",
                  a["sync"], n_frames=len(states), stop_reason=a["runlog"].stop_reason if a["runlog"] else None)
  out = Path(args.out) if args.out else None

  def compose(i):
    f = src.read_at(i)
    info.index = i
    return dash.compose(states[i], f.bgr if f else None, f.depth_m if f else None, idx.history(i), a["buf"],
                        a["pipe"].events.events, info, a["extr"].calibrated)

  if args.screenshot_at is not None:
    i = int(np.clip(np.searchsorted(idx.t, args.screenshot_at), 0, len(states) - 1))
    import cv2

    p = (out or Path(".")) / f"dashboard_t{args.screenshot_at:.1f}s.png"
    cv2.imwrite(str(p), compose(i))
    print(f"[cube-vision] screenshot {p}")
  if args.dashboard_video:
    import cv2

    p = (out or Path(".")) / "dashboard.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), src.fps / max(args.dashboard_video_stride, 1), (1680, 1000))
    for i in range(0, len(states), max(args.dashboard_video_stride, 1)):
      vw.write(compose(i))
    vw.release()
    print(f"[cube-vision] dashboard video {p}")
  _print_summary(a["summary"], out)
  if args.headless:
    return 0
  return _replay_gui(compose, info, len(states), src.fps, out)


def _replay_gui(compose, info, n, fps, out) -> int:
  import cv2

  cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
  cv2.resizeWindow(WINDOW, 1680, 1000)
  state = {"i": 0, "seek": None}
  cv2.createTrackbar("frame", WINDOW, 0, max(n - 1, 1), lambda v: state.__setitem__("seek", v))
  playing, speed = True, 1.0
  t_last = time.monotonic()
  pos = 0.0
  while True:
    if state["seek"] is not None:
      pos = float(state["seek"])
      state["seek"] = None
    now = time.monotonic()
    if playing:
      pos = min(pos + (now - t_last) * fps * speed, n - 1)
      if pos >= n - 1:
        playing = False
    t_last = now
    i = int(pos)
    info.paused, info.speed = not playing, speed
    img = compose(i)
    cv2.imshow(WINDOW, img)
    cv2.setTrackbarPos("frame", WINDOW, i)
    state["seek"] = None
    k = cv2.waitKeyEx(15)
    if k in (ord("q"), 27):
      break
    if k == ord(" "):
      playing = not playing
    elif k in (ord("."), 65363, 2555904):
      playing, pos = False, min(pos + 1, n - 1)
    elif k in (ord(","), 65361, 2424832):
      playing, pos = False, max(pos - 1, 0)
    elif k == ord("]"):
      speed = min(speed * 2, 16)
    elif k == ord("["):
      speed = max(speed / 2, 1 / 16)
    elif k == ord("s"):
      p = (out or Path(".")) / f"dashboard_frame{i:06d}.png"
      cv2.imwrite(str(p), img)
      print(f"[cube-vision] screenshot {p}")
    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
      break
  cv2.destroyAllWindows()
  return 0


# ----------------------------------------------------------------------------- live
def open_camera(args):
  if args.camera == "realsense":
    return RealSenseSource(args.realsense_serial, args.width or 848, args.height or 480, int(args.fps or 30))
  return OpenCVCameraSource(args.camera, args.width, args.height, args.fps)


def run_live(args) -> int:
  import cv2

  model = build_model(args)
  try:
    src = open_camera(args)
  except SourceUnavailable as exc:
    raise SystemExit(f"ERROR: {exc}")
  try:
    intr = resolve_intrinsics(args, src.device_intrinsics())
    intr.check_frame(*src.size)
  except IntrinsicsError as exc:
    raise SystemExit(f"ERROR: {exc}")
  extr = load_camera_to_palm(Path(args.camera_to_palm) if args.camera_to_palm else None)
  buf = TelemetryBuffer()
  udp = UdpTelemetry(args.udp, args.forward_udp) if args.udp else None
  pipe = VisionPipeline(model, intr, extr, cfg=PipelineConfig(camera_yaw_axis=_axis(args), **_guards(args)),
                        telemetry=buf)
  out = Path(args.out) if args.out else Path("outputs") / "cube_vision" / time.strftime("live_%Y%m%d_%H%M%S")
  out.mkdir(parents=True, exist_ok=True)
  if not getattr(args, "intrinsics", None) and src.device_intrinsics() is not None:
    intr.save(out / "device.intrinsics.json")  # the SDK values actually used (input for verify-device-intrinsics)
  jsonl = JsonlWriter(Path(args.jsonl) if args.jsonl else out / "poses.jsonl")
  csvw = CsvWriter(out / "poses.csv")
  rec = Recorder(out / "recording", args.fps or 30.0, src.size, intr.depth_scale_m_per_unit if src.has_depth else None) \
    if args.record else None
  dash = None if args.headless else Dashboard(model, intr, with_twin=not args.no_twin)
  info = ViewInfo("LIVE", src.label, src.has_depth, ("telemetry: " + udp.label) if udp else "telemetry: none (camera only)",
                  "sync: host receive time (camera and UDP on this host)")
  print(f"[cube-vision] LIVE {src.label}; {intr.label()}; {extr.message}; writing {out}")
  if dash is not None:
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1680, 1000)
  t_end = time.time() + args.max_seconds if args.max_seconds else None
  draw_every = 1.0 / args.display_hz if getattr(args, "display_hz", None) else 0.0
  t_drawn = -1e9
  try:
    while True:
      if udp is not None:
        for s in udp.poll():
          buf.add(s)
      f = src.read()
      if f is None:
        print("[cube-vision] camera returned no frame; stopping")
        break
      st = pipe.process(f)
      if udp is not None:
        alive = udp.last_rx_host is not None and time.time() - udp.last_rx_host < pipe.events.cfg.telemetry_timeout_s
        pipe.events.telemetry(st.t_exp, alive, "(no UDP packet; reason unknown live - see the run log / terminal)")
      r = frame_record(st)
      jsonl.write(r)
      csvw.write(r)
      if rec is not None:
        rec.write(f)
      now = time.monotonic()
      if dash is not None and now - t_drawn >= draw_every:  # drawing the panel costs ~35 ms: throttled so tracking+recording keep the camera rate
        t_drawn = now
        hist = list(pipe.hist)
        img = dash.compose(st, f.bgr, f.depth_m, hist, buf, pipe.events.events, info, extr.calibrated)
        cv2.imshow(WINDOW, img)
        k = cv2.waitKey(1)
        if k in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
          break
        if k == ord("s"):
          p = out / f"dashboard_{time.strftime('%H%M%S')}.png"
          cv2.imwrite(str(p), img)
          print(f"[cube-vision] screenshot {p}")
      if t_end and time.time() > t_end:
        break
  except KeyboardInterrupt:
    pass
  finally:
    src.close()
    jsonl.close()
    csvw.close()
    if rec is not None:
      rec.close()
    if udp is not None:
      udp.close()
    if dash is not None:
      cv2.destroyAllWindows()
  s = pipe.summary("LIVE", src.label)
  write_summary(out / "summary.json", s)
  _print_summary(s, out)
  return 0


def _print_summary(s: dict, out) -> None:
  y = s["yaw"]
  print(f"[cube-vision] frames {s['frames']}  states {json.dumps({k: round(v, 3) for k, v in s['state_fraction'].items()})}  "
        f"depth {json.dumps({k: round(v, 3) for k, v in s['depth_source_fraction'].items()})}")
  print(f"[cube-vision] rotation {y['yaw_total_deg']:+.1f} deg = {y['revolutions']:.3f} rev ({y['axis_frame']} frame, "
        f"{y['axis_source']}); excluded {y['excluded_intervals']} interval(s) {y['excluded_time_s']:.2f} s")
  if out:
    print(f"[cube-vision] outputs in {out}")
