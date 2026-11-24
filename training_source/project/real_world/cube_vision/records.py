"""Per-frame JSONL (compatible with bridge_aruco_pose.py), CSV and trial summary."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from .geometry import quat_wxyz_from_matrix


def _l(v):
  return None if v is None else [float(x) for x in np.asarray(v).reshape(-1)]


def frame_record(st) -> dict:
  """``st``: pipeline.FrameState.  Keys used by bridge_aruco_pose.convert_tracker_record:
  frame, host_time_s, tracking_valid, cube_center_m, cube_rotation_matrix,
  visible_marker_ids, reprojection_error_px."""
  tr, dr, pose = st.track, st.depth, st.track.pose
  rec = {
    "frame": int(tr.frame), "host_time_s": float(tr.host_time_s),
    "camera_time_s": tr.camera_time_s, "video_time_s": st.video_time_s,
    "tracking_valid": bool(tr.tracking_valid), "tracking_state": tr.state.value,
    "visible_marker_ids": list(tr.visible_ids), "n_visible_markers": len(tr.visible_ids),
    "unknown_marker_ids": list(tr.unknown_ids), "reprojection_error_px": None,
  }
  if pose is not None:
    pos = dr.position_m if (dr is not None and dr.position_m is not None) else pose.t
    q = quat_wxyz_from_matrix(pose.R)
    rec.update({
      "cube_center_m": _l(pos), "cube_center_pnp_m": _l(pose.t),
      "cube_rotation_matrix": np.asarray(pose.R).tolist(),
      "cube_quaternion_wxyz": _l(q), "cube_quaternion_xyzw": _l(np.roll(q, -1)),
      "inlier_marker_ids": list(pose.inlier_ids), "rejected_marker_ids": list(pose.rejected_ids),
      "layout_conflict_marker_ids": list(pose.layout_conflict_ids),
      "inlier_corner_count": 4 * len(pose.inlier_ids), "reprojection_error_px": float(pose.reproj_rms_px),
      "reprojection_error_max_px": float(pose.reproj_max_px), "pose_method": pose.method,
      "pose_ambiguous": bool(pose.ambiguous), "pose_disambiguation": pose.disambiguation,
      "ippe_ambiguity_ratio": pose.ambiguity_ratio,
      "position_std_m": _l(pose.pos_std_m), "rotation_std_deg": _l(pose.rot_std_deg),
      "pose_confidence": float(pose.confidence),
    })
  if dr is not None:
    rec.update({"depth_source": dr.source.value, "cube_depth_z_m": dr.z_m, "cube_range_m": dr.range_m,
                "depth_sigma_m": dr.sigma_range_m, "pnp_range_m": dr.pnp_range_m, "depth_stream_range_m": dr.depth_range_m,
                "depth_valid_ratio": dr.valid_ratio, "depth_valid_pixels": dr.n_valid, "depth_sampled_pixels": dr.n_sampled,
                "depth_confidence": dr.confidence, "depth_status": dr.status})
  rec.update({"intrinsics_status": st.intrinsics_status, "layout_status": st.layout_status,
              "filter_status": st.filter_status, "yaw_total_rad": st.yaw_rad, "yaw_rate_rad_s": st.yaw_rate,
              "revolutions": st.revolutions, "yaw_axis_source": st.yaw_axis_source})
  if st.palm is not None:
    rec["palm"] = {"cube_position_palm_m": _l(st.palm["p"]), "cube_quaternion_palm_wxyz": _l(st.palm["q"]),
                   "extrinsics": st.palm["file"]}
  else:
    rec["palm"] = None
    rec["palm_status"] = st.extrinsics_message
  if st.telemetry is not None:
    rec["telemetry"] = {"step": st.telemetry.packet.get("step"), "t_run_s": st.telemetry.packet.get("t"),
                        "age_s": st.telemetry_age_s}
  if st.comparison is not None:
    rec["vision_vs_proprio"] = st.comparison
  return rec


class JsonlWriter:
  def __init__(self, path: Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    self.f = open(path, "a", encoding="utf-8")

  def write(self, rec: dict) -> None:
    self.f.write(json.dumps(rec, separators=(",", ":"), default=_default) + "\n")
    self.f.flush()

  def close(self) -> None:
    self.f.close()


def _default(o):
  if isinstance(o, np.ndarray):
    return o.tolist()
  if isinstance(o, (np.floating, np.integer)):
    return o.item()
  if isinstance(o, float) and not math.isfinite(o):
    return None
  raise TypeError(type(o))


CSV_COLUMNS = ["frame", "host_time_s", "video_time_s", "tracking_state", "tracking_valid", "n_visible_markers",
               "visible_marker_ids", "reprojection_error_px", "pose_confidence", "x_m", "y_m", "z_m",
               "palm_x_m", "palm_y_m", "palm_z_m", "qw", "qx", "qy", "qz", "yaw_total_deg", "yaw_rate_deg_s",
               "revolutions", "depth_source", "cube_depth_z_mm", "cube_range_mm", "depth_valid_ratio",
               "depth_valid_pixels", "depth_confidence"]


def csv_row(rec: dict) -> dict:
  c = rec.get("cube_center_m") or [None] * 3
  q = rec.get("cube_quaternion_wxyz") or [None] * 4
  palm = (rec.get("palm") or {}).get("cube_position_palm_m") or [None] * 3
  f = lambda v, k=1.0: None if v is None else v * k  # noqa: E731
  return {"frame": rec["frame"], "host_time_s": rec["host_time_s"], "video_time_s": rec.get("video_time_s"),
          "tracking_state": rec["tracking_state"], "tracking_valid": rec["tracking_valid"],
          "n_visible_markers": rec["n_visible_markers"], "visible_marker_ids": " ".join(map(str, rec["visible_marker_ids"])),
          "reprojection_error_px": rec.get("reprojection_error_px"), "pose_confidence": rec.get("pose_confidence"),
          "x_m": c[0], "y_m": c[1], "z_m": c[2], "palm_x_m": palm[0], "palm_y_m": palm[1], "palm_z_m": palm[2],
          "qw": q[0], "qx": q[1], "qy": q[2], "qz": q[3],
          "yaw_total_deg": f(rec.get("yaw_total_rad"), 57.29577951308232),
          "yaw_rate_deg_s": f(rec.get("yaw_rate_rad_s"), 57.29577951308232), "revolutions": rec.get("revolutions"),
          "depth_source": rec.get("depth_source"), "cube_depth_z_mm": f(rec.get("cube_depth_z_m"), 1000.0),
          "cube_range_mm": f(rec.get("cube_range_m"), 1000.0), "depth_valid_ratio": rec.get("depth_valid_ratio"),
          "depth_valid_pixels": rec.get("depth_valid_pixels"), "depth_confidence": rec.get("depth_confidence")}


class CsvWriter:
  def __init__(self, path: Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    self.f = open(path, "w", newline="", encoding="utf-8")
    self.w = csv.DictWriter(self.f, CSV_COLUMNS)
    self.w.writeheader()

  def write(self, rec: dict) -> None:
    self.w.writerow(csv_row(rec))

  def close(self) -> None:
    self.f.close()


def write_summary(path_json: Path, summary: dict) -> None:
  Path(path_json).write_text(json.dumps(summary, indent=2, default=_default) + "\n", encoding="utf-8")
  md = Path(path_json).with_suffix(".md")
  y = summary.get("yaw", {})
  lines = [f"# Trial summary: {summary.get('source', '')}", "",
           f"- Mode: {summary.get('mode')}; frames {summary.get('frames')}; duration {summary.get('duration_s', 0):.1f} s",
           f"- Layout: {summary.get('layout')}; intrinsics: {summary.get('intrinsics')}",
           f"- Camera-to-palm: {summary.get('extrinsics')}",
           f"- Tracking: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in summary.get("state_fraction", {}).items()),
           f"- Depth source: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in summary.get("depth_source_fraction", {}).items()),
           f"- Rotation ({y.get('axis_frame')} frame, {y.get('axis_source')}): total {y.get('yaw_total_deg', 0):+.1f} deg = "
           f"{y.get('revolutions', 0):.3f} revolutions ({y.get('quarter_turns', 0):.2f} quarter turns), direction "
           f"{y.get('direction')}",
           f"- Excluded intervals: {y.get('excluded_intervals', 0)} ({y.get('excluded_time_s', 0):.2f} s); bridged gaps: "
           f"{y.get('bridged_intervals', 0)}",
           f"- Median reprojection error: {summary.get('median_reprojection_px')} px",
           f"- Events: {summary.get('event_counts')}", ""]
  if summary.get("comparison"):
    lines += ["## Vision vs proprioceptive estimate", "", "```", json.dumps(summary["comparison"], indent=2), "```", ""]
  lines += ["Notes: vision is an external visual measurement (quality per frame in the JSONL), the proprioceptive "
            "value is the kit's cube estimator; neither is ground truth. Motor current is measured telemetry, not force."]
  md.write_text("\n".join(lines) + "\n", encoding="utf-8")
