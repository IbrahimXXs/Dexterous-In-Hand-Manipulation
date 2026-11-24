"""Camera-to-palm extrinsics: strict loading and a measured-calibration solver.

Convention (unchanged from bridge_aruco_pose.py)::

  p_palm = R_palm_camera @ p_camera + translation_palm_camera_m

File keys: ``calibrated`` (must be true), ``translation_palm_camera_m``,
``quaternion_palm_camera_wxyz``, ``frame_convention``.  Files with ``calibrated: false``
(e.g. camera_to_palm_iphone.json, an identity placeholder) are *reported* and never used:
there is no identity fallback anywhere in this package.

Calibration (``solve_camera_to_palm``): the cube is placed at N >= 1 physically known poses
in the palm frame (a jig fixed to the palm; the palm frame is MJLab's
``palm_center_pose_w``, as in kit/hand_scene.py).  For each placement the tracker's
camera-frame pose is averaged over many frames.  The rigid transform is solved by Kabsch on
the 8 cube corners of all placements; the residuals (mm, deg) are the quality report.
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime as _dt
import json
from pathlib import Path

import numpy as np

from .geometry import project_to_so3, quat_wxyz_from_matrix, matrix_from_quat_wxyz, rotation_angle_deg


@dataclass(frozen=True)
class CameraToPalm:
  R: np.ndarray       # R_palm_camera
  t: np.ndarray       # translation_palm_camera_m
  path: str
  info: dict

  def point(self, p_cam: np.ndarray) -> np.ndarray:
    return self.R @ np.asarray(p_cam, np.float64) + self.t

  def rotation(self, R_cam_cube: np.ndarray) -> np.ndarray:
    return self.R @ np.asarray(R_cam_cube, np.float64)


@dataclass(frozen=True)
class ExtrinsicsStatus:
  calibrated: bool
  message: str
  transform: CameraToPalm | None = None


def load_camera_to_palm(path: Path | None) -> ExtrinsicsStatus:
  """Never raises for a missing/uncalibrated file: returns calibrated=False with the reason."""
  if path is None:
    return ExtrinsicsStatus(False, "no camera-to-palm file given: palm-frame pose NOT available")
  path = Path(path)
  if not path.is_file():
    return ExtrinsicsStatus(False, f"camera-to-palm file not found: {path}")
  try:
    d = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as exc:
    return ExtrinsicsStatus(False, f"cannot read {path}: {exc}")
  if d.get("calibrated") is not True:
    return ExtrinsicsStatus(False, f"{path.name} is marked calibrated={d.get('calibrated')!r} "
                                   "(placeholder) - palm-frame pose NOT calibrated")
  try:
    from ..bridge_aruco_pose import load_camera_to_palm as _bridge_load
  except ImportError:  # pragma: no cover
    from bridge_aruco_pose import load_camera_to_palm as _bridge_load
  try:
    t, R = _bridge_load(path)  # the bridge's own validation (finite, quaternion)
  except ValueError as exc:
    return ExtrinsicsStatus(False, f"{path.name}: {exc}")
  return ExtrinsicsStatus(True, f"camera-to-palm calibrated ({path.name})", CameraToPalm(R, t, str(path), d))


def kabsch(P_src: np.ndarray, P_dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """R, t minimising sum |R p_src + t - p_dst|^2 (proper rotation)."""
  P_src, P_dst = np.asarray(P_src, np.float64), np.asarray(P_dst, np.float64)
  cs, cd = P_src.mean(0), P_dst.mean(0)
  H = (P_src - cs).T @ (P_dst - cd)
  U, _, Vt = np.linalg.svd(H)
  D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
  R = Vt.T @ D @ U.T
  return R, cd - R @ cs


def average_pose(Rs: list[np.ndarray], ts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict]:
  R = project_to_so3(np.sum(np.stack(Rs), axis=0))
  t = np.mean(np.stack(ts), axis=0)
  spread_mm = float(1000 * np.sqrt(np.mean(np.sum((np.stack(ts) - t) ** 2, axis=1))))
  spread_deg = float(np.sqrt(np.mean([rotation_angle_deg(R, Ri) ** 2 for Ri in Rs])))
  return R, t, {"n": len(Rs), "position_spread_mm": spread_mm, "rotation_spread_deg": spread_deg}


def solve_camera_to_palm(samples: list[dict], cube_corners: np.ndarray) -> dict:
  """samples: [{"R_cam_cube", "t_cam_cube", "R_palm_cube", "t_palm_cube"}].  Returns R, t + residuals."""
  if not samples:
    raise ValueError("at least one reference placement is needed")
  src, dst = [], []
  for s in samples:
    src.append(cube_corners @ np.asarray(s["R_cam_cube"]).T + np.asarray(s["t_cam_cube"]))
    dst.append(cube_corners @ np.asarray(s["R_palm_cube"]).T + np.asarray(s["t_palm_cube"]))
  R, t = kabsch(np.concatenate(src), np.concatenate(dst))
  res = []
  for s, a, b in zip(samples, src, dst):
    pred_R = R @ np.asarray(s["R_cam_cube"])
    res.append({"corner_rms_mm": float(1000 * np.sqrt(np.mean(np.sum((a @ R.T + t - b) ** 2, axis=1)))),
                "center_error_mm": float(1000 * np.linalg.norm(R @ np.asarray(s["t_cam_cube"]) + t - np.asarray(s["t_palm_cube"]))),
                "rotation_error_deg": rotation_angle_deg(pred_R, np.asarray(s["R_palm_cube"]))})
  return {"R_palm_camera": R, "translation_palm_camera_m": t, "residuals": res,
          "max_center_error_mm": max(r["center_error_mm"] for r in res),
          "max_rotation_error_deg": max(r["rotation_error_deg"] for r in res)}


def save_camera_to_palm(path: Path, sol: dict, *, reference: str, n_samples: int, accept: bool,
                        thresholds: dict | None = None) -> None:
  """Writes calibrated=true only when ``accept`` (residual thresholds met); otherwise false + the result."""
  q = quat_wxyz_from_matrix(sol["R_palm_camera"])
  payload = {
    "calibrated": bool(accept),
    "frame_convention": "p_palm = R_palm_camera @ p_camera + translation_palm_camera_m",
    "translation_palm_camera_m": [float(v) for v in sol["translation_palm_camera_m"]],
    "quaternion_palm_camera_wxyz": [float(v) for v in q],
    "method": "cube placed at known palm-frame poses; Kabsch on cube corners (real_world/cube_vision/extrinsics.py)",
    "reference": reference, "n_placements": n_samples, "residuals": sol["residuals"],
    "max_center_error_mm": sol["max_center_error_mm"], "max_rotation_error_deg": sol["max_rotation_error_deg"],
    "acceptance_thresholds": thresholds or {},
    "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
  }
  if not accept:
    payload["warning"] = "Residuals above the acceptance thresholds: NOT marked calibrated. Re-check the jig and placements."
  Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_reference_poses(path: Path) -> list[dict]:
  """Reference file: {"reference": "...", "placements": [{"name", "cube_position_palm_m", "cube_quaternion_palm_wxyz"}]}."""
  d = json.loads(Path(path).read_text(encoding="utf-8"))
  out = []
  for p in d.get("placements", []):
    out.append({"name": p.get("name", f"p{len(out)}"), "t_palm_cube": np.asarray(p["cube_position_palm_m"], np.float64),
                "R_palm_cube": matrix_from_quat_wxyz(p["cube_quaternion_palm_wxyz"])})
  if not out:
    raise ValueError(f"{path}: no placements")
  return out
