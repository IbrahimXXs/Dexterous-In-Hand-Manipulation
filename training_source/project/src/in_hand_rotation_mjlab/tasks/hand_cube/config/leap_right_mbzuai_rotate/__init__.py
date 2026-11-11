"""Registration for continuous MBZUAI cube Z rotation on the flat right hand."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import leap_right_mbzuai_rotate_env_cfg
from .rl_cfg import leap_right_mbzuai_rotate_ppo_cfg


register_mjlab_task(
  task_id="Mjlab-Leap-Right-HandCube-MBZUAI-Rotate",
  env_cfg=leap_right_mbzuai_rotate_env_cfg(),
  play_env_cfg=leap_right_mbzuai_rotate_env_cfg(play=True),
  rl_cfg=leap_right_mbzuai_rotate_ppo_cfg(),
)
