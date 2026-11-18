"""Task ids of the hardware-limit A/C study (experiments/hwlimit_ac).

=====================  ===========================================================
task id                actor input
=====================  ===========================================================
Mjlab-Leap-HwLim-A     320-D joint history (baseline)
Mjlab-Leap-HwLim-C     320 + 19 features from the frozen estimator B
=====================  ===========================================================

Both use ``env_cfgs.hwlimit_env_cfg`` (study environment + approved joint
command limits + feasible grasp cache) and ``rl_cfg.hwlimit_ppo_cfg``.  The
original tasks, including ``Mjlab-Leap-Scratch-*``, are not modified.
"""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import hwlimit_env_cfg
from .rl_cfg import hwlimit_ppo_cfg

for _cond in ("A", "C"):
  register_mjlab_task(
    task_id=f"Mjlab-Leap-HwLim-{_cond}",
    env_cfg=hwlimit_env_cfg(),
    play_env_cfg=hwlimit_env_cfg(play=True),
    rl_cfg=hwlimit_ppo_cfg(_cond),
  )
