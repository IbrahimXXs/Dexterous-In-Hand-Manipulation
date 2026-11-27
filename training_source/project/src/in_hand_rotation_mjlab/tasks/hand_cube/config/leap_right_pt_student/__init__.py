"""Task id of the PT-student teacher (experiments/pt_student, PLAN.md section 1).

``Mjlab-Leap-PT-Teacher``: the ``Mjlab-Leap-GRV2-C5`` environment unchanged (rewards, yaw clip, curriculum,
terminations, domain randomisation, observation noise/delay, resets, limits, action), and C5's PPO configuration
(``hwlimit_ppo_cfg("A")``: 3001 updates, zero-initialised actor output) with ONE change: the actor reads the
observation groups ``actor`` (C5's 320-D proprioception) and ``critic`` (83-D privileged state: clean joint
positions, velocities, command error, previous command, palm-frame cube pose and velocities, cube size, mass,
centre-of-mass offset, friction).  The critic is unchanged.  Logged under ``logs/rsl_rl/pt_student_teacher``.
No existing task is modified.
"""

from mjlab.tasks.registry import register_mjlab_task

from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_grasp_rotation_v2.env_cfgs import grv2_env_cfg
from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_hwlimit_study.rl_cfg import hwlimit_ppo_cfg

TEACHER_ACTOR_GROUPS = ("actor", "critic")


def teacher_rl_cfg():
  cfg = hwlimit_ppo_cfg("A")
  cfg.experiment_name = "pt_student_teacher"
  cfg.obs_groups = {"actor": TEACHER_ACTOR_GROUPS, "critic": ("critic",)}
  return cfg


register_mjlab_task(task_id="Mjlab-Leap-PT-Teacher", env_cfg=grv2_env_cfg("C5"), play_env_cfg=grv2_env_cfg("C5", play=True),
                    rl_cfg=teacher_rl_cfg())
