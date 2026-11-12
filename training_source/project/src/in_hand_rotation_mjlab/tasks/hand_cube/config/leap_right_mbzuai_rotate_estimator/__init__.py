"""Additive task variants for the sensorless cube-state estimation study.

The original ``Mjlab-Leap-Right-HandCube-MBZUAI-Rotate`` registration is not
touched.  All variants share the same environment (corrected reset anchor,
see ``estimation/env_fixes.py``), reward, critic and PPO hyper-parameters, and
differ only in the actor:

=====================================================  =======================
task id                                                actor input
=====================================================  =======================
Mjlab-Leap-Right-HandCube-MBZUAI-Rotate-FixedReset      320 (fine-tuning control)
Mjlab-Leap-Right-HandCube-MBZUAI-Rotate-Estimator       320 + 19 estimated state
Mjlab-Leap-Right-HandCube-MBZUAI-Rotate-EstContact      320 + 4 estimated contacts
=====================================================  =======================
"""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import fixed_reset_env_cfg
from .rl_cfg import estimator_ppo_cfg

register_mjlab_task(
  task_id="Mjlab-Leap-Right-HandCube-MBZUAI-Rotate-FixedReset",
  env_cfg=fixed_reset_env_cfg(),
  play_env_cfg=fixed_reset_env_cfg(play=True),
  rl_cfg=estimator_ppo_cfg(feature_mode=None),
)
register_mjlab_task(
  task_id="Mjlab-Leap-Right-HandCube-MBZUAI-Rotate-Estimator",
  env_cfg=fixed_reset_env_cfg(),
  play_env_cfg=fixed_reset_env_cfg(play=True),
  rl_cfg=estimator_ppo_cfg(feature_mode="full"),
)
register_mjlab_task(
  task_id="Mjlab-Leap-Right-HandCube-MBZUAI-Rotate-EstContact",
  env_cfg=fixed_reset_env_cfg(),
  play_env_cfg=fixed_reset_env_cfg(play=True),
  rl_cfg=estimator_ppo_cfg(feature_mode="contact"),
)
