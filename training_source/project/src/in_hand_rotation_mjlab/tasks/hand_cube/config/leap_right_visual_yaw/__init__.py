"""Registration for flat MBZUAI visual Z-yaw control on the physical right hand."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  leap_right_visual_yaw_env_cfg,
)
from .rl_cfg import (
  leap_right_visual_yaw_ppo_cfg,
)


register_mjlab_task(
  task_id="Mjlab-Leap-Right-HandCube-MBZUAI-Visual-Yaw",
  env_cfg=leap_right_visual_yaw_env_cfg(),
  play_env_cfg=leap_right_visual_yaw_env_cfg(play=True),
  rl_cfg=leap_right_visual_yaw_ppo_cfg(),
)
