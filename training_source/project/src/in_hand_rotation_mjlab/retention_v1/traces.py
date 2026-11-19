"""Per-step trajectory recording for the Retention-V1 diagnosis (simulation only).

``record_traces`` rolls a policy out deterministically from a synchronised reset and
records, for every environment, the FIRST episode only (until its first termination;
later steps are masked by ``alive``).  The episode timeout is pushed beyond the horizon
(``episode_length_s = (num_steps + 20) * dt``), exactly as the study evaluator does for its
30-s protocol, so an automatic timeout reset can never be counted as rotation.

Recorded per step (palm-centre frame of ``cube_pose_in_palm_frame``):
  cube_pos_palm [3], cube_quat_palm [4], drift (m, from the episode-initial world
  position, the termination's measure), tilt (rad, physical, from the initial attitude),
  yaw increment (genuine, rad), cube lin/ang velocity in the palm frame [3+3],
  fingertip-pad contact flags [4: if, mf, rf, th], total normal force on each distal
  link averaged over the control interval [4: th, if, mf, rf] (force study's passive
  sensor), fingertip-tip positions relative to the
  cube centre in the palm frame [4x3], joint positions [16], joint commands [16],
  command-at-approved-limit flags [16], command headroom to the nearest approved limit
  [16], executed normalised action tanh(u) [16], and per-step termination codes.
Per environment: reset-cache grasp id, domain-randomisation values, first-termination
step and reason.

All of this is simulator ground truth used for analysis only; nothing is fed to a policy.
"""

from __future__ import annotations

import math

import numpy as np
import torch

TERM_CODES = {"time_out": 1, "cube_fell": 2, "cube_too_fast": 3, "cube_pose_deviation": 4, "nan": 5}
TIP_GEOMS = ("if_tip", "mf_tip", "rf_tip", "th_tip")


def make_env_with_force_sensor(task: str, num_envs: int, seed: int, device: str, *, preset: str = "nominal",
                               episode_length_s: float | None = None):
  """``hwlimit_study.evaluation.make_env`` plus the force study's passive distal-link contact
  sensor (labels only; proven not to change the physics, tests/test_force_estimation.py)."""
  from in_hand_rotation_mjlab.estimation.sim import _import_tasks
  from in_hand_rotation_mjlab.force_estimation.labels import force_sensor_cfg
  from in_hand_rotation_mjlab.hwlimit_study.evaluation import PRESETS
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.noise import UniformNoiseCfg

  _import_tasks()
  env_cfg = load_env_cfg(task)
  env_cfg.scene.num_envs = int(num_envs)
  env_cfg.seed = int(seed)
  env_cfg.curriculum = {}
  if episode_length_s is not None:
    env_cfg.episode_length_s = float(episode_length_s)
  p = PRESETS[preset]
  for name, params in p.get("events", {}).items():
    env_cfg.events[name].params.update(params)
  if "obs_noise" in p:
    n = p["obs_noise"]
    env_cfg.observations["actor"].terms["joint_pos"].noise = UniformNoiseCfg(n_min=-n, n_max=n)
  env_cfg.scene.sensors = tuple(env_cfg.scene.sensors or ()) + (force_sensor_cfg(env_cfg.decimation),)
  agent_cfg = load_rl_cfg(task)
  raw = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  return RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions), raw, agent_cfg


@torch.no_grad()
def record_traces(task: str, checkpoint: str, seed: int, num_envs: int, num_steps: int, device: str = "cuda:0",
                  preset: str = "nominal", estimator=None) -> dict[str, np.ndarray]:
  """``estimator``: optional estimator B; its palm-frame position and contact-probability
  estimates on the actor's own observation are recorded (for sim-vs-hardware comparison
  through the same estimator the kit's monitor uses).  Never fed to the policy."""
  from in_hand_rotation_mjlab.estimation.sim import GroundTruth
  from in_hand_rotation_mjlab.hwlimit_study.evaluation import load_study_policy, raw_dt
  from in_hand_rotation_mjlab.scratch_study.geometry import physical_tilt, world_up_in_body, yaw_increment
  from in_hand_rotation_mjlab.tasks.hand_cube.mdp.observations import palm_center_pose_w
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from mjlab.utils.lab_api.math import quat_apply, quat_inv

  env, raw, agent_cfg = make_env_with_force_sensor(task, num_envs, seed, device, preset=preset,
                                                   episode_length_s=(num_steps + 20) * raw_dt())
  actor = load_study_policy(env, agent_cfg, checkpoint, device)
  from in_hand_rotation_mjlab.force_estimation.labels import ForceLabeler, raw_other_hand_cube_normal_force

  labeler = ForceLabeler(raw)
  cube, robot = raw.scene["cube"], raw.scene["robot"]
  term = raw.action_manager.get_term("joint_pos")
  lo, hi = term.approved_limits()
  tm = raw.termination_manager
  hand_cfg = SceneEntityCfg("robot", body_names=("palm",))
  hand_cfg.resolve(raw.scene)
  tip_ids = [robot.find_geoms(g, preserve_order=True)[0][0] for g in TIP_GEOMS]
  n, T = num_envs, num_steps

  def buf(*shape, dtype=np.float32):
    return np.zeros((T, n, *shape), dtype=dtype)

  out = {k: buf(*s) for k, s in {
    "cube_pos_palm": (3,), "cube_quat_palm": (4,), "lin_vel_palm": (3,), "ang_vel_palm": (3,),
    "tip_rel_cube_palm": (4, 3), "joint_pos": (16,), "command": (16,), "headroom": (16,), "action": (16,)}.items()}
  for k in ("drift", "tilt", "yaw_inc"):
    out[k] = buf()
  out["contact"] = buf(4, dtype=np.uint8)
  out["distal_force"] = buf(4)  # total normal force on each distal link [N], order th, if, mf, rf
  out["other_force"] = buf()
  if estimator is not None:
    out["est_pos_palm"] = buf(3)
    out["est_contact_prob"] = buf(4)  # normal force on the cube from all other hand links (palm, phalanges) [N]
  out["at_limit"] = buf(16, dtype=np.uint8)
  out["term_code"] = buf(dtype=np.int8)
  out["alive"] = buf(dtype=np.uint8)

  env.reset()
  obs = env.get_observations()
  cache_ids = raw._hand_cube_reset_cache_ids.clone().cpu().numpy()
  q_prev = cube.data.root_link_quat_w.clone()
  pos0 = cube.data.root_link_pos_w.clone()
  up0 = world_up_in_body(q_prev)
  first = torch.ones(n, dtype=torch.bool, device=device)
  active = torch.ones(n, dtype=torch.bool, device=device)
  end_step = torch.full((n,), -1, dtype=torch.long, device=device)
  end_code = torch.zeros(n, dtype=torch.int8, device=device)
  dr = None
  for t in range(T):
    u = actor(obs)
    a_t = torch.tanh(u)
    obs, _, dones, _ = env.step(u)
    dones = dones.bool()
    if dr is None:
      dr = {k: v.clone().cpu().numpy() for k, v in GroundTruth(raw).randomization().items()}
    q = cube.data.root_link_quat_w
    pos = cube.data.root_link_pos_w
    if bool(first.any()):  # anchor on the first post-reset step (as the evaluator / termination)
      pos0[first] = pos[first]
      up0[first] = world_up_in_body(q[first])
    dpsi = yaw_increment(q, q_prev)
    live = active & ~dones & ~first
    cs = obs["cube_state"]
    palm_c, palm_q = palm_center_pose_w(env=raw, hand_cfg=hand_cfg, palm_center_geom_expr="palm_collision_.*")
    qi = quat_inv(palm_q)
    tips_w = robot.data.geom_pos_w[:, tip_ids]  # [n,4,3]
    rel = tips_w - pos[:, None, :]
    rel_palm = quat_apply(qi[:, None, :].expand(-1, 4, -1).reshape(-1, 4), rel.reshape(-1, 3)).reshape(n, 4, 3)
    cmd = term._target
    code = torch.zeros(n, dtype=torch.int8, device=device)
    if bool(dones.any()):
      for name, c in TERM_CODES.items():
        if name in tm.active_terms:
          code = torch.where(tm.get_term(name).bool(), torch.full_like(code, c), code)
    newly = active & dones
    end_step = torch.where(newly, torch.full_like(end_step, t), end_step)
    end_code = torch.where(newly, code, end_code)
    rec = {
      "cube_pos_palm": cs[:, 0:3], "cube_quat_palm": cs[:, 3:7], "lin_vel_palm": cs[:, 7:10], "ang_vel_palm": cs[:, 10:13],
      "tip_rel_cube_palm": rel_palm, "joint_pos": robot.data.joint_pos, "command": cmd,
      "headroom": torch.minimum(cmd - lo, hi - cmd), "action": a_t,
      "drift": (pos - pos0).norm(dim=-1), "tilt": physical_tilt(q, up_ref=up0), "yaw_inc": torch.where(live, dpsi, 0.0),
    }
    for k, v in rec.items():
      out[k][t] = v.float().cpu().numpy()
    out["contact"][t] = (cs[:, 13:17] > 0.5).cpu().numpy()
    out["distal_force"][t] = labeler.interval()["mean"].cpu().numpy()
    out["other_force"][t] = raw_other_hand_cube_normal_force(raw).cpu().numpy()
    if estimator is not None:
      pe = estimator(obs["actor"])
      out["est_pos_palm"][t] = pe["pos"].float().cpu().numpy()
      out["est_contact_prob"][t] = pe["contact_prob"].float().cpu().numpy()
    out["at_limit"][t] = (((cmd - lo).abs() < 1e-4) | ((hi - cmd).abs() < 1e-4)).cpu().numpy()
    out["term_code"][t] = torch.where(newly, code, torch.zeros_like(code)).cpu().numpy()
    out["alive"][t] = (active & ~dones).cpu().numpy()
    first = dones.clone()
    q_prev = q.clone()
    active &= ~dones
  env.close()
  out["cache_id"] = cache_ids
  out["end_step"] = end_step.cpu().numpy()
  out["end_code"] = end_code.cpu().numpy()
  out["dt"] = np.float32(raw_dt())
  for k, v in (dr or {}).items():
    out[k] = v
  return out


def turns_until(trace: dict, steps: int) -> np.ndarray:
  """Uninterrupted turns of the first episode within the first ``steps`` control steps."""
  return trace["yaw_inc"][:steps].sum(0) / (2 * math.pi)
