"""Camera / recording sources with explicit timestamps.  A requested source that is not
available raises ``SourceUnavailable`` with the reason - never a silent fallback.

* ``OpenCVCameraSource``  UVC webcam index (``0``) or a stream URL (e.g. an iPhone streaming
  app exposing MJPEG/RTSP).  RGB only; metric range then comes from PnP.
* ``RealSenseSource``     Intel RealSense via pyrealsense2: colour + depth aligned to colour
  (``rs.align``), SDK colour intrinsics (status device_factory_unverified) and depth scale.
* ``RecordingSource``     RGB video + optional metric depth (a directory of 16-bit PNGs or an
  .npz with ``depth`` [N,H,W] uint16) + optional per-frame timestamps CSV.  Random access
  for scrubbing.  A colourised depth *video* is not metric and is refused.
* ``Recorder``            writes rgb.mp4 + frames.csv (+ depth/*.png16 + depth_meta.json).

Timestamps: ``host_time_s`` = this host's wall clock at capture (time.time()); the camera's
own clock, when the device exposes one, goes in ``camera_time_s``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np

from .depth import depth_to_meters
from .intrinsics import STATUS_DEVICE, CameraIntrinsics


class SourceUnavailable(RuntimeError):
  pass


@dataclass
class Frame:
  index: int
  host_time_s: float
  camera_time_s: float | None
  bgr: np.ndarray
  depth_m: np.ndarray | None = None    # aligned to bgr, metres, NaN = invalid
  depth_raw: np.ndarray | None = None  # sensor units (uint16) for recording


class OpenCVCameraSource:
  kind = "opencv"
  has_depth = False

  def __init__(self, device: str, width: int | None = None, height: int | None = None, fps: float | None = None):
    import cv2

    self.cv2 = cv2
    dev = int(device) if str(device).isdigit() else device
    self.cap = cv2.VideoCapture(dev)
    if not self.cap.isOpened():
      raise SourceUnavailable(f"camera {device!r} could not be opened (no device / no permission / wrong URL). "
                              "List devices with: scripts/cube_vision.py list-cameras")
    if width:
      self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
      self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
      self.cap.set(cv2.CAP_PROP_FPS, fps)
    ok, img = self.cap.read()
    if not ok or img is None:
      raise SourceUnavailable(f"camera {device!r} opened but delivered no frame")
    self._first = (img, time.time())  # the startup-check frame is delivered by the first read()
    self.size = (img.shape[1], img.shape[0])
    self.label = f"OpenCV camera {device} {self.size[0]}x{self.size[1]}"
    self.camera_id = f"opencv:{device}"
    self.i = 0

  def device_intrinsics(self) -> CameraIntrinsics | None:
    return None  # a UVC device reports none; calibrate it

  def read(self) -> Frame | None:
    if self._first is not None:
      (img, t), self._first = self._first, None
      ok = True
    else:
      ok, img = self.cap.read()
      t = time.time()
    if not ok:
      return None
    ms = self.cap.get(self.cv2.CAP_PROP_POS_MSEC)
    f = Frame(self.i, t, (ms / 1000.0) if ms and ms > 0 else None, img)
    self.i += 1
    return f

  def close(self) -> None:
    self.cap.release()


class RealSenseSource:
  kind = "realsense"
  has_depth = True

  def __init__(self, serial: str | None = None, width: int = 848, height: int = 480, fps: int = 30):
    try:
      import pyrealsense2 as rs
    except ImportError as exc:
      raise SourceUnavailable("pyrealsense2 is not installed (uv sync --extra vision --extra realsense)") from exc
    self.rs = rs
    ctx = rs.context()
    devs = list(ctx.query_devices())
    if not devs:
      raise SourceUnavailable("no Intel RealSense device found (check USB3 cable / permissions / udev rules)")
    cfg = rs.config()
    if serial:
      cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    self.pipe = rs.pipeline()
    try:
      prof = self.pipe.start(cfg)
    except RuntimeError as exc:
      raise SourceUnavailable(f"RealSense stream {width}x{height}@{fps} could not start: {exc}") from exc
    dev = prof.get_device()
    self.depth_scale = float(dev.first_depth_sensor().get_depth_scale())
    self.align = rs.align(rs.stream.color)
    vp = prof.get_stream(rs.stream.color).as_video_stream_profile()
    ci = vp.get_intrinsics()
    self.size = (ci.width, ci.height)
    name = dev.get_info(rs.camera_info.name)
    sn = dev.get_info(rs.camera_info.serial_number)
    self.camera_id = f"realsense:{name}:{sn}"
    K = np.array([[ci.fx, 0, ci.ppx], [0, ci.fy, ci.ppy], [0, 0, 1.0]])
    dist = np.asarray(list(ci.coeffs)[:5], np.float64)
    self._intr = CameraIntrinsics(K, dist, self.size, STATUS_DEVICE, self.camera_id, None, self.depth_scale, True,
                                  extra={"rs_distortion_model": str(ci.model)})
    self.label = f"RealSense {name} SN {sn} {self.size[0]}x{self.size[1]}@{fps} (depth aligned to colour)"
    self.i = 0

  def device_intrinsics(self) -> CameraIntrinsics:
    return self._intr

  def read(self) -> Frame | None:
    frames = self.pipe.wait_for_frames(2000)
    t = time.time()
    al = self.align.process(frames)
    c, d = al.get_color_frame(), al.get_depth_frame()
    if not c or not d:
      return None
    img = np.asanyarray(c.get_data()).copy()
    raw = np.asanyarray(d.get_data()).copy()
    f = Frame(self.i, t, c.get_timestamp() / 1000.0, img, depth_to_meters(raw, self.depth_scale), raw)
    self.i += 1
    return f

  def close(self) -> None:
    self.pipe.stop()


class RecordingSource:
  kind = "recording"

  def __init__(self, video: Path, depth: Path | None = None, timestamps: Path | None = None, depth_scale: float | None = None):
    import cv2

    self.cv2 = cv2
    self.video = Path(video)
    if not self.video.is_file():
      raise SourceUnavailable(f"video not found: {video}")
    self.cap = cv2.VideoCapture(str(self.video))
    if not self.cap.isOpened():
      raise SourceUnavailable(f"cannot decode {video}")
    self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
    self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    self._next = 0
    self.timestamps = _read_timestamps(timestamps) if timestamps else None
    if self.timestamps is None and (self.video.parent / "frames.csv").is_file() and timestamps is None:
      self.timestamps = _read_timestamps(self.video.parent / "frames.csv")
    self.depth_kind, self._depth, self.depth_scale = None, None, depth_scale
    if depth is not None:
      self._open_depth(Path(depth))
    self.has_depth = self._depth is not None
    self.label = f"REPLAY {self.video.name} ({self.n} frames @ {self.fps:.2f} fps" + \
      (f", depth {self.depth_kind}" if self.has_depth else ", no depth") + ")"
    self.camera_id = f"recording:{self.video.name}"

  def _open_depth(self, p: Path) -> None:
    if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
      raise SourceUnavailable(f"{p.name}: a depth *video* is colour-mapped and not metric; record 16-bit PNGs or .npz")
    if p.is_dir():
      files = sorted(p.glob("*.png"))
      if not files:
        raise SourceUnavailable(f"no depth PNGs in {p}")
      meta = p / "depth_meta.json"
      if self.depth_scale is None and meta.is_file():
        self.depth_scale = float(json.loads(meta.read_text())["depth_scale_m_per_unit"])
      if self.depth_scale is None:
        raise SourceUnavailable(f"{p}: depth scale unknown (depth_meta.json or --depth-scale)")
      self._depth = {int(f.stem.split("_")[-1]): f for f in files}
      self.depth_kind = "png16"
    elif p.suffix == ".npz":
      d = np.load(p, allow_pickle=False)
      if "depth" not in d.files:
        raise SourceUnavailable(f"{p}: no 'depth' array")
      arr = d["depth"]
      if self.depth_scale is None:
        if "depth_scale_m_per_unit" in d.files:
          self.depth_scale = float(d["depth_scale_m_per_unit"])
        elif np.issubdtype(arr.dtype, np.integer):
          raise SourceUnavailable(f"{p}: integer depth needs depth_scale_m_per_unit")
      idx = d["frame_index"] if "frame_index" in d.files else np.arange(len(arr))
      self._depth = {int(i): k for k, i in enumerate(idx)}
      self._depth_arr = arr
      self.depth_kind = "npz"
    else:
      raise SourceUnavailable(f"unsupported depth file {p}")

  def _depth_for(self, i: int):
    if self._depth is None or i not in self._depth:
      return None, None
    if self.depth_kind == "png16":
      raw = self.cv2.imread(str(self._depth[i]), self.cv2.IMREAD_UNCHANGED)
    else:
      raw = self._depth_arr[self._depth[i]]
    if raw is None:
      return None, None
    if raw.shape[:2] != (self.size[1], self.size[0]):
      raise SourceUnavailable(f"depth frame {i} is {raw.shape[1]}x{raw.shape[0]}, colour is {self.size}: not aligned")
    return depth_to_meters(raw, self.depth_scale), raw

  def time_of(self, i: int) -> tuple[float, float | None]:
    if self.timestamps is not None and i in self.timestamps:
      return self.timestamps[i]
    return i / self.fps, None

  def read_at(self, i: int) -> Frame | None:
    if i < 0 or i >= self.n:
      return None
    if i != self._next:
      self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, i)
    ok, img = self.cap.read()
    if not ok:
      return None
    self._next = i + 1
    ht, ct = self.time_of(i)
    dm, raw = self._depth_for(i)
    return Frame(i, ht, ct, img, dm, raw)

  def read(self) -> Frame | None:
    return self.read_at(self._next)

  def device_intrinsics(self):
    return None

  def close(self) -> None:
    self.cap.release()


def _read_timestamps(path: Path) -> dict[int, tuple[float, float | None]]:
  out = {}
  with open(path, newline="") as f:
    for row in csv.DictReader(f):
      ct = row.get("camera_time_s")
      out[int(row["frame"])] = (float(row["host_time_s"]), float(ct) if ct not in (None, "", "nan") else None)
  return out


class Recorder:
  def __init__(self, out_dir: Path, fps: float, size: tuple[int, int], depth_scale: float | None):
    import cv2

    self.cv2 = cv2
    self.dir = Path(out_dir)
    self.dir.mkdir(parents=True, exist_ok=False)
    self.vw = cv2.VideoWriter(str(self.dir / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    self.csv = open(self.dir / "frames.csv", "w", newline="")
    self.w = csv.writer(self.csv)
    self.w.writerow(["frame", "host_time_s", "camera_time_s"])
    self.depth_scale = depth_scale
    if depth_scale is not None:
      (self.dir / "depth").mkdir()
      (self.dir / "depth" / "depth_meta.json").write_text(json.dumps({"depth_scale_m_per_unit": depth_scale,
                                                                      "aligned_to_color": True}))
    self.k = 0

  def write(self, f: Frame) -> None:
    self.vw.write(f.bgr)
    self.w.writerow([self.k, f"{f.host_time_s:.6f}", "" if f.camera_time_s is None else f"{f.camera_time_s:.6f}"])
    if self.depth_scale is not None and f.depth_raw is not None:
      self.cv2.imwrite(str(self.dir / "depth" / f"depth_{self.k:06d}.png"), f.depth_raw.astype(np.uint16))
    self.k += 1

  def close(self) -> None:
    self.vw.release()
    self.csv.close()


def list_cameras(max_index: int = 8) -> dict:
  import cv2

  import platform

  devs = sorted(str(p) for p in Path("/dev").glob("video*")) if Path("/dev").is_dir() else []
  out = {"opencv": [], "realsense": [], "video_devices": devs}
  if platform.system() == "Linux":
    indices = sorted(int(d[len("/dev/video"):]) for d in devs if d[len("/dev/video"):].isdigit())
  else:
    indices = list(range(max_index))
  try:
    prev = cv2.utils.logging.getLogLevel()
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
  except AttributeError:
    prev = None
  for i in indices:
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
      ok, img = cap.read()
      out["opencv"].append({"index": i, "frame": None if not ok else [img.shape[1], img.shape[0]]})
    cap.release()
  if prev is not None:
    cv2.utils.logging.setLogLevel(prev)
  try:
    import pyrealsense2 as rs

    for d in rs.context().query_devices():
      out["realsense"].append({"name": d.get_info(rs.camera_info.name), "serial": d.get_info(rs.camera_info.serial_number)})
  except ImportError:
    out["realsense"] = "pyrealsense2 not installed"
  return out
