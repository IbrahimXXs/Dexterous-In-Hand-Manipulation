"""Task id of the PT-student teacher (experiments/pt_student, PLAN.md section 1).

``Mjlab-Leap-PT-Teacher``: the ``Mjlab-Leap-GRV2-C5`` environment unchanged (rewards, yaw clip, curriculum,
terminations, domain randomisation, observation noise/delay, resets, limits, action), and C5's PPO configuration
(``hwlimit_ppo_cfg("A")``: 3001 updates, zero-initialised actor output) with ONE change: the actor reads the
observation groups ``actor`` (C5's 320-D proprioception) and ``critic`` (83-D privileged state: clean joint
positions, velocities, command error, previous command, palm-frame cube pose and velocities, cube size, mass,
centre-of-mass offset, friction).  The critic is unchanged.  Logged under ``logs/rsl_rl/pt_student_teacher``.
No existing task is modified.

``Mjlab-Leap-PT-Teacher-AR`` (optional variant, PLAN.md section 4.3): the same teacher with ONE added reward term,
``action_rate_squashed`` = weight -0.075 x sum_j (tanh u_t - tanh u_{t-1})^2.  Logged under
``logs/rsl_rl/pt_student_teacher_ar``.
"""

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.registry import register_mjlab_task

from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_grasp_rotation_v2.env_cfgs import grv2_env_cfg
from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_hwlimit_study.rl_cfg import hwlimit_ppo_cfg
from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_pt_student.rewards import ACTION_RATE_WEIGHT, squashed_action_rate_l2

TEACHER_ACTOR_GROUPS = ("actor", "critic")


def teacher_rl_cfg(experiment_name: str = "pt_student_teacher"):
  cfg = hwlimit_ppo_cfg("A")
  cfg.experiment_name = experiment_name
  cfg.obs_groups = {"actor": TEACHER_ACTOR_GROUPS, "critic": ("critic",)}
  return cfg


def teacher_ar_env_cfg(play: bool = False):
  cfg = grv2_env_cfg("C5", play=play)
  cfg.rewards["action_rate_squashed"] = RewardTermCfg(func=squashed_action_rate_l2, weight=ACTION_RATE_WEIGHT, params={})
  return cfg


register_mjlab_task(task_id="Mjlab-Leap-PT-Teacher", env_cfg=grv2_env_cfg("C5"), play_env_cfg=grv2_env_cfg("C5", play=True),
                    rl_cfg=teacher_rl_cfg())
register_mjlab_task(task_id="Mjlab-Leap-PT-Teacher-AR", env_cfg=teacher_ar_env_cfg(), play_env_cfg=teacher_ar_env_cfg(play=True),
                    rl_cfg=teacher_rl_cfg("pt_student_teacher_ar"))
