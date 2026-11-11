"""Registration for Playground-style reorientation on the physical right hand."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  leap_right_playground_reorient_env_cfg,
  leap_right_playground_reorient_tilted_env_cfg,
)
from .rl_cfg import (
  leap_right_playground_reorient_ppo_cfg,
  leap_right_playground_reorient_tilted_ppo_cfg,
)


register_mjlab_task(
  task_id="Mjlab-Leap-Right-HandCube-Playground-Reorient",
  env_cfg=leap_right_playground_reorient_env_cfg(),
  play_env_cfg=leap_right_playground_reorient_env_cfg(play=True),
  rl_cfg=leap_right_playground_reorient_ppo_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Leap-Right-HandCube-Playground-Reorient-Tilted",
  env_cfg=leap_right_playground_reorient_tilted_env_cfg(),
  play_env_cfg=leap_right_playground_reorient_tilted_env_cfg(play=True),
  rl_cfg=leap_right_playground_reorient_tilted_ppo_cfg(),
)
