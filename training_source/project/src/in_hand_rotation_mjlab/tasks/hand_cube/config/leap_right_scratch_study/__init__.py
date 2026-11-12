"""Task ids of the from-scratch sensorless state-estimation study.

=================================  ==========================================
task id                            actor input
=================================  ==========================================
Mjlab-Leap-Scratch-A               320-D joint history (baseline)
Mjlab-Leap-Scratch-C               320 + 19 features from the frozen estimator
Mjlab-Leap-Scratch-D               320 + 19 features from ground truth (oracle)
Mjlab-Leap-Scratch-A-ClipAction    pilot only: A with the original clipped actions
=================================  ==========================================

All share the corrected environment of ``env_cfgs.scratch_env_cfg`` and the
PPO configuration of ``rl_cfg.scratch_ppo_cfg``.  The original tasks are not
modified.
"""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import scratch_env_cfg
from .rl_cfg import scratch_ppo_cfg

for _cond in ("A", "C", "D"):
  register_mjlab_task(
    task_id=f"Mjlab-Leap-Scratch-{_cond}",
    env_cfg=scratch_env_cfg(),
    play_env_cfg=scratch_env_cfg(play=True),
    rl_cfg=scratch_ppo_cfg(_cond),
  )

register_mjlab_task(
  task_id="Mjlab-Leap-Scratch-A-ClipAction",
  env_cfg=scratch_env_cfg(squash="clip"),
  play_env_cfg=scratch_env_cfg(play=True, squash="clip"),
  rl_cfg=scratch_ppo_cfg("A", squash="clip"),
)
