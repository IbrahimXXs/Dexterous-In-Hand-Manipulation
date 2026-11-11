#!/usr/bin/env python3
"""Real-time pose decomposition monitor for iPhone ArUco cube tracking.

Reads the tracker JSONL stream and decomposes each observation into three
separate coordinate frames:

  1. **Device AR pose** (T_world_device)
      – iPhone 6-DoF pose in the AR world frame. Available via asMagic's
        device_transform when ARKit is active. Falls back to identity if
        the stream does not provide it.

  2. **Observed marker pose** (T_camera_cube)
      – The raw ArUco-based cube pose in the iPhone camera optical frame
        (X-right, Y-down, Z-forward). This comes directly from the
        CubePoseEstimator via solvePnP on detected marker corners.

  3. **Object pose in hand frame** (T_palm_cube)
      – Cube pose expressed in the LEAP palm coordinate frame. Computed
        as: T_palm_cube = T_palm_camera @ T_camera_cube
        where T_palm_camera is the extrinsic calibration.

Usage:
  python monitor_pose_decomposition.py \\
    --tracker-jsonl data/iphone_runs/<stamp>/tracker.jsonl \\
    --camera-to-palm real_world/camera_to_palm_iphone.json \\
    [--output-csv data/iphone_runs/<stamp>/poses_decomposed.csv]

When --live is passed, the monitor tails the JSONL and prints to stdout
in real time. Otherwise, it processes the full file and writes CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Quaternion / rotation utilities (self-contained, no external imports)
# ---------------------------------------------------------------------------

def _rotation_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _quat_wxyz_from_rotation(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = np.array([0.25*s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s])
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = 2.0 * np.sqrt(1 + R[0,0] - R[1,1] - R[2,2])
            q = np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
        elif i == 1:
            s = 2.0 * np.sqrt(1 + R[1,1] - R[0,0] - R[2,2])
            q = np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
        else:
            s = 2.0 * np.sqrt(1 + R[2,2] - R[0,0] - R[1,1])
            q = np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])
    q = q / np.linalg.norm(q)
    return -q if q[0] < 0 else q


def _euler_xyz_deg(R: np.ndarray) -> tuple[float, float, float]:
    """Intrinsic XYZ Euler angles in degrees from a rotation matrix."""
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        rx = np.arctan2(R[2,1], R[2,2])
        ry = np.arctan2(-R[2,0], sy)
        rz = np.arctan2(R[1,0], R[0,0])
    else:
        rx = np.arctan2(-R[1,2], R[1,1])
        ry = np.arctan2(-R[2,0], sy)
        rz = 0.0
    return (np.degrees(rx), np.degrees(ry), np.degrees(rz))


def _make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def _invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

def decompose_record(
    record: dict[str, Any],
    T_palm_camera: np.ndarray,
) -> dict[str, Any] | None:
    """Decompose a single tracker JSONL record into three poses."""

    if not record.get("tracking_valid", False):
        return None

    # ── 1. Device AR pose (T_world_device) ─────────────────────────
    # asMagic may provide device_transform as a 4x4 column-major matrix.
    # If not present, we record identity (stationary device assumption).
    device_transform = record.get("device_transform")
    if device_transform is not None:
        T_world_device = np.asarray(device_transform, dtype=np.float64).reshape(4, 4)
    else:
        T_world_device = np.eye(4, dtype=np.float64)

    # ── 2. Observed marker pose (T_camera_cube) ───────────────────
    R_camera_cube = np.asarray(record["cube_rotation_matrix"], dtype=np.float64)
    t_camera_cube = np.asarray(record["cube_center_m"], dtype=np.float64)
    T_camera_cube = _make_T(R_camera_cube, t_camera_cube)

    # ── 3. Object pose in hand frame (T_palm_cube) ────────────────
    T_palm_cube = T_palm_camera @ T_camera_cube

    # Extract readable components
    R_palm_cube = T_palm_cube[:3, :3]
    t_palm_cube = T_palm_cube[:3, 3]

    return {
        "frame": record.get("frame", -1),
        "host_time_s": record.get("host_time_s", 0.0),
        "visible_markers": record.get("visible_marker_ids", []),
        "reproj_error_px": record.get("reprojection_error_px"),
        # Device AR pose
        "device_pos_x_m": float(T_world_device[0, 3]),
        "device_pos_y_m": float(T_world_device[1, 3]),
        "device_pos_z_m": float(T_world_device[2, 3]),
        "device_euler_x_deg": _euler_xyz_deg(T_world_device[:3, :3])[0],
        "device_euler_y_deg": _euler_xyz_deg(T_world_device[:3, :3])[1],
        "device_euler_z_deg": _euler_xyz_deg(T_world_device[:3, :3])[2],
        # Camera-frame cube pose
        "cam_cube_x_m": float(t_camera_cube[0]),
        "cam_cube_y_m": float(t_camera_cube[1]),
        "cam_cube_z_m": float(t_camera_cube[2]),
        "cam_cube_euler_x_deg": _euler_xyz_deg(R_camera_cube)[0],
        "cam_cube_euler_y_deg": _euler_xyz_deg(R_camera_cube)[1],
        "cam_cube_euler_z_deg": _euler_xyz_deg(R_camera_cube)[2],
        "cam_cube_quat_wxyz": _quat_wxyz_from_rotation(R_camera_cube).tolist(),
        # Palm-frame cube pose (= hand-frame object pose)
        "palm_cube_x_m": float(t_palm_cube[0]),
        "palm_cube_y_m": float(t_palm_cube[1]),
        "palm_cube_z_m": float(t_palm_cube[2]),
        "palm_cube_euler_x_deg": _euler_xyz_deg(R_palm_cube)[0],
        "palm_cube_euler_y_deg": _euler_xyz_deg(R_palm_cube)[1],
        "palm_cube_euler_z_deg": _euler_xyz_deg(R_palm_cube)[2],
        "palm_cube_quat_wxyz": _quat_wxyz_from_rotation(R_palm_cube).tolist(),
        # Rotation progress (angular distance from identity)
        "rotation_angle_deg": float(np.degrees(
            np.arccos(np.clip((np.trace(R_palm_cube) - 1.0) / 2.0, -1.0, 1.0))
        )),
    }


def load_camera_to_palm(path: Path) -> np.ndarray:
    """Load T_palm_camera from the calibration JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    t = np.asarray(payload["translation_palm_camera_m"], dtype=np.float64)
    q = np.asarray(payload["quaternion_palm_camera_wxyz"], dtype=np.float64)
    R = _rotation_from_quat_wxyz(q)
    return _make_T(R, t)


def process_batch(
    jsonl_path: Path,
    T_palm_camera: np.ndarray,
    output_csv: Path | None,
) -> list[dict[str, Any]]:
    """Process a complete JSONL file."""
    results = []
    with jsonl_path.open("r") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            decomposed = decompose_record(record, T_palm_camera)
            if decomposed is not None:
                results.append(decomposed)

    if output_csv and results:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = list(results[0].keys())
        with output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"Wrote {len(results)} decomposed poses → {output_csv}")

    return results


def tail_live(
    jsonl_path: Path,
    T_palm_camera: np.ndarray,
    print_every: int = 5,
) -> None:
    """Tail a growing JSONL and print decomposed poses live."""
    frame_count = 0
    with jsonl_path.open("r") as f:
        # Seek to end for live tailing
        f.seek(0, os.SEEK_END)
        print(
            "Live monitoring... (Ctrl+C to stop)\n"
            "┌─────────────────────────────────────────────────────────────┐\n"
            "│ Frame  │ Markers │ Camera XYZ (cm)  │ Palm XYZ (cm)  │ Rot°│\n"
            "├─────────────────────────────────────────────────────────────┤",
            flush=True,
        )
        try:
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.03)
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                decomposed = decompose_record(record, T_palm_camera)
                if decomposed is None:
                    continue
                frame_count += 1
                if frame_count % print_every == 0:
                    cx = decomposed["cam_cube_x_m"] * 100
                    cy = decomposed["cam_cube_y_m"] * 100
                    cz = decomposed["cam_cube_z_m"] * 100
                    px = decomposed["palm_cube_x_m"] * 100
                    py = decomposed["palm_cube_y_m"] * 100
                    pz = decomposed["palm_cube_z_m"] * 100
                    rot = decomposed["rotation_angle_deg"]
                    ids = decomposed["visible_markers"]
                    print(
                        f"│ {decomposed['frame']:6d} │ {str(ids):>7s} │ "
                        f"{cx:+5.1f},{cy:+5.1f},{cz:+5.1f} │ "
                        f"{px:+5.1f},{py:+5.1f},{pz:+5.1f} │ "
                        f"{rot:4.0f} │",
                        flush=True,
                    )
        except KeyboardInterrupt:
            print(
                "\n└─────────────────────────────────────────────────────────────┘"
            )
            print(f"Monitored {frame_count} valid frames.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracker-jsonl", type=Path, required=True,
                        help="Path to the tracker JSONL output")
    parser.add_argument("--camera-to-palm", type=Path, required=True,
                        help="Camera-to-palm calibration JSON")
    parser.add_argument("--output-csv", type=Path, default=None,
                        help="Write batch-processed results to CSV")
    parser.add_argument("--live", action="store_true",
                        help="Tail the JSONL in real time instead of batch processing")
    parser.add_argument("--print-every", type=int, default=5,
                        help="In live mode, print every N-th valid frame")
    args = parser.parse_args()

    T_palm_camera = load_camera_to_palm(args.camera_to_palm)
    print(f"T_palm_camera loaded from {args.camera_to_palm}")
    print(f"  translation = {T_palm_camera[:3, 3].tolist()}")
    print(f"  rotation det = {np.linalg.det(T_palm_camera[:3, :3]):.6f}")
    print()

    if args.live:
        tail_live(args.tracker_jsonl, T_palm_camera, args.print_every)
    else:
        results = process_batch(args.tracker_jsonl, T_palm_camera, args.output_csv)
        if results:
            print(f"\nSummary ({len(results)} valid poses):")
            angles = [r["rotation_angle_deg"] for r in results]
            print(f"  Rotation range: {min(angles):.1f}° – {max(angles):.1f}°")
            distances = [np.sqrt(r["cam_cube_x_m"]**2 + r["cam_cube_y_m"]**2 + r["cam_cube_z_m"]**2) * 100
                         for r in results]
            print(f"  Camera distance: {min(distances):.1f} – {max(distances):.1f} cm")


if __name__ == "__main__":
    main()
