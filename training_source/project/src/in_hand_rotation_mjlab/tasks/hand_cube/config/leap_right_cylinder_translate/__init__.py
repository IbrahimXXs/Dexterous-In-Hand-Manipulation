"""Register Task 1 autonomous table-pickup tasks."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  leap_right_hand_cylinder_pickup_env_cfg,
  leap_right_hand_cylinder_pickup_translate_env_cfg,
)
from .rl_cfg import (
  leap_right_hand_cylinder_pickup_ppo_cfg,
  leap_right_hand_cylinder_pickup_translate_ppo_cfg,
)


register_mjlab_task(
  task_id="Mjlab-Leap-Right-HandCylinder-Pickup",
  env_cfg=leap_right_hand_cylinder_pickup_env_cfg(),
  play_env_cfg=leap_right_hand_cylinder_pickup_env_cfg(play=True),
  rl_cfg=leap_right_hand_cylinder_pickup_ppo_cfg(),
)


register_mjlab_task(
  task_id="Mjlab-Leap-Right-HandCylinder-Pickup-Translate",
  env_cfg=leap_right_hand_cylinder_pickup_translate_env_cfg(),
  play_env_cfg=leap_right_hand_cylinder_pickup_translate_env_cfg(play=True),
  rl_cfg=leap_right_hand_cylinder_pickup_translate_ppo_cfg(),
)
