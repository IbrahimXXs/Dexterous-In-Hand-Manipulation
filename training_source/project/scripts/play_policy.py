"""MJLab's stock `play` viewer with this project's hand-cube tasks registered.

mjlab's own `play` entry point only knows mjlab's built-in tasks; importing
in_hand_rotation_mjlab.tasks first adds Mjlab-Leap-Right-HandCube-* to the registry.

To look like the reference video (reference/continuous_z_model400_preview_30s.mp4,
made by record_policy_video.py --mbzuai-cube --seed 11), the cube gets the same
MBZUAI texture (visual only; physics unchanged) and the environment seed defaults to 11.

  python scripts/play_policy.py Mjlab-Leap-Right-HandCube-MBZUAI-Rotate \
      --checkpoint-file /abs/path/model_400.pt --device cpu --viewer native

Environment variables:
  PLAY_SEED=<int>       environment seed (default 11, the reference video's seed)
  PLAY_PLAIN_CUBE=1     keep MJLab's untextured cube
  PLAY_DEBUG_VIS=1      show the debug axes (frame_viz overlay + mjlab's world frame)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import in_hand_rotation_mjlab.tasks  # noqa: F401,E402  (registers the tasks)
import mjlab.scripts.play as mjlab_play  # noqa: E402
from record_policy_video import add_mbzuai_cube_visual  # noqa: E402

_original_load_env_cfg = mjlab_play.load_env_cfg


def _load_env_cfg(task_id, *args, **kwargs):
  cfg = _original_load_env_cfg(task_id, *args, **kwargs)
  cfg.seed = int(os.environ.get("PLAY_SEED", "11"))
  if os.environ.get("PLAY_PLAIN_CUBE") != "1" and "cube" in cfg.scene.entities:
    add_mbzuai_cube_visual(cfg)
  # The frame_viz command only draws debug axes (world/palm/cube frames and a
  # palm->cube arrow) in the native viewer; the reference video has none.
  if os.environ.get("PLAY_DEBUG_VIS") != "1":
    for command in (cfg.commands or {}).values():
      if hasattr(command, "debug_vis"):
        command.debug_vis = False
  return cfg


mjlab_play.load_env_cfg = _load_env_cfg

# mjlab's native viewer hard-codes opt.frame = mjFRAME_WORLD, which draws large
# RGB world axes at the origin (under the hand). The reference video has none.
if os.environ.get("PLAY_DEBUG_VIS") != "1":
  import mujoco  # noqa: E402
  from mjlab.viewer.native.viewer import NativeMujocoViewer  # noqa: E402

  _original_setup_camera = NativeMujocoViewer._setup_camera

  def _setup_camera_without_world_frame(self) -> None:
    _original_setup_camera(self)
    self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE.value

  NativeMujocoViewer._setup_camera = _setup_camera_without_world_frame

if __name__ == "__main__":
  mjlab_play.main()
