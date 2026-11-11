"""Registration for the flat LEAP pinch-lift-jiggle-yaw task."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import leap_right_palm_lift_yaw_env_cfg
from .rl_cfg import leap_right_palm_lift_yaw_ppo_cfg


register_mjlab_task(
  task_id="Mjlab-Leap-Right-HandCube-Palm-Lift-Yaw",
  env_cfg=leap_right_palm_lift_yaw_env_cfg(),
  play_env_cfg=leap_right_palm_lift_yaw_env_cfg(play=True),
  rl_cfg=leap_right_palm_lift_yaw_ppo_cfg(),
)
