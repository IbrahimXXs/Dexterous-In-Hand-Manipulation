"""Register hardware-aligned 60-second right-hand cube rotation."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import leap_right_hardware_60s_env_cfg
from .rl_cfg import leap_right_hardware_60s_ppo_cfg


register_mjlab_task(
    task_id="Mjlab-Leap-Right-HandCube-Rotate-Hardware60s",
    env_cfg=leap_right_hardware_60s_env_cfg(),
    play_env_cfg=leap_right_hardware_60s_env_cfg(play=True),
    rl_cfg=leap_right_hardware_60s_ppo_cfg(),
)
