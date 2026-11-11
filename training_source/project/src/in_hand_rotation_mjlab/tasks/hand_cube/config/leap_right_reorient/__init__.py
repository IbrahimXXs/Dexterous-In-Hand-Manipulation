"""Task registration for arbitrary right-hand cube reorientation."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import leap_right_hand_cube_reorient_env_cfg
from .rl_cfg import leap_right_hand_cube_reorient_ppo_cfg


register_mjlab_task(
  task_id="Mjlab-Leap-Right-HandCube-Reorient",
  env_cfg=leap_right_hand_cube_reorient_env_cfg(),
  play_env_cfg=leap_right_hand_cube_reorient_env_cfg(play=True),
  rl_cfg=leap_right_hand_cube_reorient_ppo_cfg(),
)

