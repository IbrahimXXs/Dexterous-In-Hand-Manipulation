#!/usr/bin/env python3
"""Run the exported continuous positive-yaw policy in native MuJoCo."""

from __future__ import annotations

import argparse
import csv
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

from policy.yaw_policy import ContinuousYawPolicy

MODEL = ROOT / "models" / "scene_right_mbzuai_cube.xml"
WEIGHTS = ROOT / "checkpoint" / "policy_weights.npz"
CACHE = (
    ROOT
    / "training_source"
    / "project"
    / "src"
    / "in_hand_rotation_mjlab"
    / "tasks"
    / "hand_cube"
    / "cache"
    / "leap_right_mbzuai_58mm_policy_grasp_cache.npz"
)
JOINTS = (
    "if_mcp", "if_rot", "if_pip", "if_dip",
    "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
    "th_cmc", "th_axl", "th_mcp", "th_ipl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--policy-hz", type=float, default=20.0)
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument(
        "--grasp-index",
        type=int,
        default=312,
        help="Deterministic retained-cube state from the original task cache",
    )
    parser.add_argument("--no-viewer", action="store_true", help="Run headless for testing or remote machines")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "simulation")
    return parser.parse_args()


def yaw_from_wxyz(q: np.ndarray) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def joint_addresses(model: mujoco.MjModel) -> np.ndarray:
    addresses = []
    for name in JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"Model is missing policy joint {name!r}")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(addresses, dtype=np.int64)


def run(args: argparse.Namespace) -> dict[str, float | int | str]:
    if args.duration_s <= 0 or args.policy_hz <= 0 or args.playback_speed <= 0:
        raise ValueError("duration, policy rate, and playback speed must be positive")

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    # Match the original task's 5 ms physics step and ten-step decimation.
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    policy = ContinuousYawPolicy(WEIGHTS)
    qadr = joint_addresses(model)
    cube_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    cube_qadr = int(model.jnt_qposadr[cube_joint])

    cache = np.load(CACHE)
    if not 0 <= args.grasp_index < cache["joint_pos"].shape[0]:
        raise ValueError(f"grasp index must be in [0, {cache['joint_pos'].shape[0] - 1}]")
    measured_q = cache["joint_pos"][args.grasp_index].astype(np.float32)
    initial_target = cache["joint_pos_target"][args.grasp_index].astype(np.float32)
    # The final Class 01 policy API resets from measured hardware state. The
    # cached simulation additionally restores the position-controller target.
    initial = policy.reset(measured_q)
    policy._prev_command = np.clip(
        initial_target, policy.lower_limits, policy.upper_limits
    ).copy()
    initial = policy._prev_command.copy()
    data.qpos[qadr] = measured_q
    data.ctrl[:] = initial
    data.qpos[cube_qadr:cube_qadr + 7] = cache["cube_pose_rel"][args.grasp_index]
    mujoco.mj_forward(model, data)

    policy_period = 1.0 / args.policy_hz
    next_policy_time = 0.0
    next_render_time = 0.0
    yaw_samples: list[float] = []
    rows: list[tuple[float, float, float, float, float]] = []
    viewer = None

    if not args.no_viewer:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)
        viewer.cam.lookat[:] = (-0.03, 0.02, 0.10)
        viewer.cam.distance = 0.48
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -18.0

    wall_start = time.perf_counter()
    try:
        while data.time < args.duration_s and (viewer is None or viewer.is_running()):
            if data.time + 1e-12 >= next_policy_time:
                command, _, _ = policy.step(data.qpos[qadr])
                data.ctrl[:] = command
                next_policy_time += policy_period

            mujoco.mj_step(model, data)
            cube_pos = data.qpos[cube_qadr:cube_qadr + 3]
            yaw_samples.append(yaw_from_wxyz(data.qpos[cube_qadr + 3:cube_qadr + 7]))
            rows.append((data.time, yaw_samples[-1], float(cube_pos[0]), float(cube_pos[1]), float(cube_pos[2])))

            if viewer is not None and data.time >= next_render_time:
                viewer.sync()
                next_render_time += 1.0 / 60.0
                target_wall = wall_start + data.time / args.playback_speed
                delay = target_wall - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
    finally:
        if viewer is not None:
            viewer.close()

    unwrapped = np.unwrap(np.asarray(yaw_samples, dtype=np.float64))
    yaw_change = float(unwrapped[-1] - unwrapped[0])
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "yaw_trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "cube_yaw_wrapped_rad", "cube_x_m", "cube_y_m", "cube_z_m"))
        writer.writerows(rows)

    summary: dict[str, float | int | str] = {
        "task": "continuous_positive_z_yaw_rotation",
        "duration_s": float(data.time),
        "policy_steps": int(math.ceil(data.time * args.policy_hz)),
        "grasp_cache_index": int(args.grasp_index),
        "yaw_change_rad": yaw_change,
        "yaw_turns": yaw_change / (2.0 * math.pi),
        "minimum_cube_height_m": min(row[4] for row in rows),
        "cube_retained": bool(min(row[4] for row in rows) > 0.10),
        "trace_csv": str(trace_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["cube_retained"]:
        raise RuntimeError(
            "cube dropped below the retained-object threshold; "
            "do not report this run as a working yaw demonstration"
        )
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

