#!/usr/bin/env python3
"""Smoothly replay the exact stable seed-11 MJLab model-400 trajectory."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.run_yaw_rotation import MODEL, joint_addresses, yaw_from_wxyz

TRAJECTORY = ROOT / "reference" / "exact_mjlab_seed11_state_trajectory.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument("--render-fps", type=float, default=60.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
        help="Replay starting from this timestamp in seconds (e.g. 5.0 to skip the initial grasp transient)",
    )
    return parser.parse_args()


def normalized_lerp_wxyz(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    if float(np.dot(a, b)) < 0.0:
        b = -b
    q = (1.0 - alpha) * a + alpha * b
    return q / np.linalg.norm(q)


def trajectory_summary(cube_pose: np.ndarray, step_dt: float) -> dict[str, object]:
    wrapped = np.asarray([yaw_from_wxyz(q[3:7]) for q in cube_pose])
    unwrapped = np.unwrap(wrapped)
    yaw_change = float(unwrapped[-1] - unwrapped[0])
    return {
        "source": "exact_mjlab_model_400_seed_11_state_replay",
        "task": "Mjlab-Leap-Right-HandCube-MBZUAI-Rotate",
        "frames": int(cube_pose.shape[0]),
        "duration_s": float((cube_pose.shape[0] - 1) * step_dt),
        "yaw_change_rad": yaw_change,
        "yaw_turns": yaw_change / (2.0 * math.pi),
        "minimum_cube_height_m": float(np.min(cube_pose[:, 2])),
        "cube_retained": bool(np.min(cube_pose[:, 2]) > 0.10),
    }


def main() -> None:
    args = parse_args()
    if args.playback_speed <= 0 or args.render_fps <= 0:
        raise ValueError("playback speed and render FPS must be positive")

    archive = np.load(TRAJECTORY)
    joint_pos = archive["joint_pos_rad"].astype(np.float64)
    cube_pose = archive["cube_pose_wxyz"].astype(np.float64)
    step_dt = float(archive["step_dt_s"])

    if args.start_time > 0:
        start_frame = min(int(args.start_time / step_dt), joint_pos.shape[0] - 2)
        joint_pos = joint_pos[start_frame:]
        cube_pose = cube_pose[start_frame:]

    summary = trajectory_summary(cube_pose, step_dt)
    if args.start_time > 0:
        summary["start_time_s"] = float(args.start_time)
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["cube_retained"]:
        raise RuntimeError("reference trajectory does not retain the cube")
    if args.no_viewer:
        return

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    qadr = joint_addresses(model)
    cube_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    cube_qadr = int(model.jnt_qposadr[cube_joint])

    # Pre-initialize data to first frame so the viewer opens at the correct pose
    data.qpos[qadr] = joint_pos[0]
    data.qpos[cube_qadr:cube_qadr + 3] = cube_pose[0, :3]
    data.qpos[cube_qadr + 3:cube_qadr + 7] = cube_pose[0, 3:7]
    mujoco.mj_forward(model, data)

    from mujoco import viewer as mj_viewer

    viewer = mj_viewer.launch_passive(model, data)
    viewer.cam.lookat[:] = (-0.03, 0.02, 0.22)
    viewer.cam.distance = 0.48
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -18.0

    duration = float(summary["duration_s"])
    render_period = 1.0 / args.render_fps
    try:
        while viewer.is_running():
            wall_start = time.perf_counter()
            while viewer.is_running():
                elapsed = (time.perf_counter() - wall_start) * args.playback_speed
                if elapsed >= duration:
                    break
                frame = elapsed / step_dt
                i0 = min(int(frame), joint_pos.shape[0] - 2)
                alpha = frame - i0
                q = (1.0 - alpha) * joint_pos[i0] + alpha * joint_pos[i0 + 1]
                cube_xyz = (1.0 - alpha) * cube_pose[i0, :3] + alpha * cube_pose[i0 + 1, :3]
                cube_quat = normalized_lerp_wxyz(cube_pose[i0, 3:7], cube_pose[i0 + 1, 3:7], alpha)
                data.qpos[qadr] = q
                data.qpos[cube_qadr:cube_qadr + 3] = cube_xyz
                data.qpos[cube_qadr + 3:cube_qadr + 7] = cube_quat
                data.time = elapsed
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(render_period)
            if not args.loop:
                break
    finally:
        viewer.close()


if __name__ == "__main__":
    main()

