#!/usr/bin/env python3
"""Record a fixed-target hold from one contact-bearing grasp-cache entry."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import cv2
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

import mjlab.tasks  # noqa: F401
import in_hand_rotation_mjlab.tasks  # noqa: F401


def _write_single_entry_cache(source: Path, output: Path, index: int) -> list[int]:
  with np.load(source, allow_pickle=False) as cache:
    sample_count = int(cache["cube_size"].shape[0])
    if not 0 <= index < sample_count:
      raise IndexError(f"cache index {index} is outside [0, {sample_count})")
    contacts = np.asarray(cache["fingertip_contact"][index], dtype=np.uint8)
    payload: dict[str, np.ndarray] = {}
    for name in cache.files:
      value = np.asarray(cache[name])
      if name in {
        "cube_size",
        "joint_pos",
        "joint_pos_target",
        "cube_pose_rel",
        "fingertip_contact",
      }:
        value = value[index : index + 1]
      payload[name] = value
  np.savez_compressed(output, **payload)
  return contacts.tolist()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--task", default="Mjlab-Leap-Right-HandCube-Playground-Reorient"
  )
  parser.add_argument("--cache-file", required=True)
  parser.add_argument("--cache-index", type=int, default=132)
  parser.add_argument("--duration-s", type=float, default=10.0)
  parser.add_argument("--width", type=int, default=960)
  parser.add_argument("--height", type=int, default=720)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--seed", type=int, default=61)
  parser.add_argument("--output-dir", required=True)
  args = parser.parse_args()

  cache_file = Path(args.cache_file).expanduser().resolve()
  if not cache_file.is_file():
    raise FileNotFoundError(cache_file)
  if args.duration_s <= 0.0:
    raise ValueError("duration-s must be positive")

  output_dir = Path(args.output_dir).expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  configure_torch_backends()

  with tempfile.TemporaryDirectory(prefix="mbzuai_cache_video_") as temp_dir:
    single_cache = Path(temp_dir) / "selected_cache_entry.npz"
    contacts = _write_single_entry_cache(
      cache_file, single_cache, args.cache_index
    )

    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args.seed
    env_cfg.viewer.width = args.width
    env_cfg.viewer.height = args.height
    env_cfg.viewer.env_idx = 0
    env_cfg.viewer.distance = 0.42
    env_cfg.viewer.azimuth = 135.0
    env_cfg.viewer.elevation = -22.0
    env_cfg.scene.entities.pop("goal_cube", None)
    env_cfg.scene.entities.pop("tracked_cube", None)
    env_cfg.events["reset_from_grasp_cache"].params["cache_file"] = str(
      single_cache
    )
    step_dt = env_cfg.sim.mujoco.timestep * env_cfg.decimation
    video_steps = round(args.duration_s / step_dt)
    env_cfg.episode_length_s = max(
      env_cfg.episode_length_s, args.duration_s + 1.0
    )
    env_cfg.terminations["time_out"] = None

    raw_env = ManagerBasedRlEnv(
      cfg=env_cfg, device=args.device, render_mode="rgb_array"
    )
    video_path = output_dir / "mbzuai_58mm_cache_validation.mp4"
    writer = cv2.VideoWriter(
      str(video_path),
      cv2.VideoWriter_fourcc(*"mp4v"),
      1.0 / step_dt,
      (args.width, args.height),
    )
    if not writer.isOpened():
      raw_env.close()
      raise RuntimeError("OpenCV could not initialize the MP4 video writer")
    action = torch.zeros(
      (1, raw_env.action_manager.total_action_dim), device=raw_env.device
    )
    found = raw_env.scene["fingertip_cube_contact"].data.found
    assert found is not None
    print(
      f"cache_index={args.cache_index} stored_contacts={contacts} "
      f"initial_sim_contacts={(found[0] > 0).to(torch.uint8).tolist()}"
    )
    retained = True
    with torch.inference_mode():
      for step in range(video_steps):
        _, _, terminated, time_out, _ = raw_env.step(action)
        if bool(terminated[0] or time_out[0]):
          retained = False
        found = raw_env.scene["fingertip_cube_contact"].data.found
        assert found is not None
        contacts_now = (found[0] > 0).to(torch.uint8).tolist()
        frame = raw_env.render()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.rectangle(frame_bgr, (12, 12), (535, 76), (20, 20, 20), -1)
        cv2.putText(
          frame_bgr,
          "58 mm / 85 g MBZUAI cache hold validation",
          (24, 38),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.65,
          (255, 255, 255),
          2,
        )
        cv2.putText(
          frame_bgr,
          f"t={(step + 1) * step_dt:4.1f}s  contacts I/M/R/T={contacts_now}",
          (24, 64),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.55,
          (120, 255, 120),
          2,
        )
        writer.write(frame_bgr)
        if (step + 1) % 50 == 0:
          height = float(raw_env.scene["cube"].data.root_link_pos_w[0, 2])
          print(
            f"step={step + 1}/{video_steps} cube_height_m={height:.4f} "
            f"contacts={contacts_now}"
          )
    writer.release()
    raw_env.close()
    print(f"retained_without_termination={retained}")
    print(f"video={video_path}")


if __name__ == "__main__":
  main()
