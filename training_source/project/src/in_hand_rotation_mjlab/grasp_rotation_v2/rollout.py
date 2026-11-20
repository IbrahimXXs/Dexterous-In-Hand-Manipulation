"""Lean first-episode rollouts with a controlled initial grasp (simulation only).

``assigned_reset`` wraps the task's ``reset_from_grasp_cache`` event.  When the raw
environment carries ``_gr2_assignment`` (a LongTensor [num_envs] of cache indices), the
grasp of every reset environment is taken from it instead of being drawn uniformly; the
original function does all the work (it draws the grasp with exactly one ``torch.randint``
call, which is answered from the assignment for the duration of the call).  Without the
attribute the wrapper is the original event.  ``_gr2_pose_range`` optionally adds the
original event's own ``pose_range`` cube perturbation (world-frame x/y/z in m,
roll/pitch/yaw in rad).  Nothing here reaches the actor: it only selects the reset state.

``rollout`` evaluates one policy (or the zero-action "hold" policy) for ``num_steps``
control steps from a synchronised reset, first episode only (the episode timeout is pushed
beyond the horizon, as in ``hwlimit_study.evaluation.evaluate``), with the same metric
definitions as that evaluator (turns = genuine yaw while alive, anchor on the first
post-reset step, survival = alive steps x dt).  Per step it keeps only compact signals
(float16) plus every reward term as PPO sees it (weight x raw x dt), so 4096 environments x
60 s fit in memory.  All recorded quantities are simulator ground truth for analysis.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from in_hand_rotation_mjlab.tasks.hand_cube.mdp import events as _events

TERM_CODES = {"time_out": 1, "cube_fell": 2, "cube_too_fast": 3, "cube_pose_deviation": 4, "nan": 5}
TIP_GEOMS = ("if_tip", "mf_tip", "rf_tip", "th_tip")
# hwlimit A seed 1 penalty-curriculum weights at update 3000 (as retention_v1/scripts/reward_audit.py)
FINAL_CURRICULUM = {"object_linvel_penalty": -0.095, "pose_diff_penalty": -0.0317, "torque_penalty": -0.317,
                    "work_penalty": -0.0317}


def assigned_reset(env, env_ids, **params):
  assign = getattr(env, "_gr2_assignment", None)
  extra = getattr(env, "_gr2_pose_range", None)
  if extra is not None:
    params = {**params, "pose_range": extra}
  if assign is None:
    return _events.reset_from_grasp_cache(env, env_ids, **params)
  ids = _events._resolve_env_ids(env, env_ids)
  want = assign.to(env.device)[ids]
  orig, calls = torch.randint, []

  def answer(high, size, *a, **k):
    assert tuple(size) == (len(ids),), (size, len(ids))
    calls.append(int(high))
    return want.clone()

  torch.randint = answer
  try:
    _events.reset_from_grasp_cache(env, env_ids, **params)
  finally:
    torch.randint = orig
  assert len(calls) == 1 and int(want.max()) < calls[0], calls
  assert torch.equal(env._hand_cube_reset_cache_ids[ids], want)


def make_env(task: str, num_envs: int, seed: int, device: str, *, preset: str = "nominal", num_steps: int = 1200,
             curriculum_weights: dict | None = FINAL_CURRICULUM, grasp_cache: str | None = None,
             terminations_off: tuple[str, ...] = ()):
  """``hwlimit_study.evaluation.make_env`` with the assignable reset, the episode timeout
  beyond the horizon and the penalty weights frozen (``curriculum_weights``).  Physics,
  observations, actions and domain randomisation are the task's own."""
  from in_hand_rotation_mjlab.estimation.sim import _import_tasks
  from in_hand_rotation_mjlab.hwlimit_study.evaluation import PRESETS, raw_dt
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.noise import UniformNoiseCfg

  _import_tasks()
  cfg = load_env_cfg(task)
  cfg.scene.num_envs, cfg.seed, cfg.curriculum = int(num_envs), int(seed), {}
  cfg.episode_length_s = (num_steps + 20) * raw_dt()
  for k, w in (curriculum_weights or {}).items():
    cfg.rewards[k].weight = w
  p = PRESETS[preset]
  for name, params in p.get("events", {}).items():
    cfg.events[name].params.update(params)
  if "obs_noise" in p:
    n = p["obs_noise"]
    cfg.observations["actor"].terms["joint_pos"].noise = UniformNoiseCfg(n_min=-n, n_max=n)
  ev = cfg.events["reset_from_grasp_cache"]
  assert ev.func is _events.reset_from_grasp_cache and ev.params.get("scale_list") is None, ev
  ev.func = assigned_reset
  if grasp_cache is not None:
    ev.params["cache_file"] = str(grasp_cache)
  for name in terminations_off:  # diagnostic only (e.g. how episodes continue without the pose-deviation rule)
    del cfg.terminations[name]
  if "cube_pose_deviation" in terminations_off and "grasp_loss" in cfg.rewards:
    del cfg.rewards["grasp_loss"]  # reads that termination; a reward term, no effect on the rollout
  agent_cfg = load_rl_cfg(task)
  raw = ManagerBasedRlEnv(cfg=cfg, device=device)
  return RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions), raw, agent_cfg


@torch.no_grad()
def rollout(task: str, checkpoint: str | None, seed: int, num_envs: int, num_steps: int, device: str = "cuda:0", *,
            preset: str = "nominal", assignment: np.ndarray | None = None, pose_range: dict | None = None,
            record_actions: bool = False, curriculum_weights: dict | None = FINAL_CURRICULUM,
            terminations_off: tuple[str, ...] = (), record_quat: bool = False) -> dict[str, np.ndarray]:
  """``checkpoint=None`` = hold policy (zero action: the command stays at the grasp target)."""
  from in_hand_rotation_mjlab.estimation.sim import GroundTruth
  from in_hand_rotation_mjlab.hwlimit_study.evaluation import load_study_policy
  from in_hand_rotation_mjlab.scratch_study.geometry import physical_tilt, world_up_in_body, yaw_increment
  from in_hand_rotation_mjlab.tasks.hand_cube.mdp.observations import palm_center_pose_w
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from mjlab.utils.lab_api.math import quat_apply, quat_inv

  env, raw, agent_cfg = make_env(task, num_envs, seed, device, preset=preset, num_steps=num_steps,
                                 curriculum_weights=curriculum_weights, terminations_off=terminations_off)
  if assignment is not None:
    raw._gr2_assignment = torch.as_tensor(np.asarray(assignment), dtype=torch.long, device=device)
  if pose_range is not None:
    raw._gr2_pose_range = dict(pose_range)
  actor = load_study_policy(env, agent_cfg, checkpoint, device) if checkpoint else None
  cube, robot = raw.scene["cube"], raw.scene["robot"]
  term = raw.action_manager.get_term("joint_pos")
  lo, hi = term.approved_limits()
  tm, rm = raw.termination_manager, raw.reward_manager
  rnames = list(rm._term_names)
  hand_cfg = SceneEntityCfg("robot", body_names=("palm",))
  hand_cfg.resolve(raw.scene)
  tip_ids = [robot.find_geoms(g, preserve_order=True)[0][0] for g in TIP_GEOMS]
  n, T, dt = num_envs, num_steps, float(raw.step_dt)

  f16 = lambda *s: np.zeros((T, n, *s), dtype=np.float16)  # noqa: E731
  out = {"yaw_inc": np.zeros((T, n), np.float32), "drift": f16(), "tilt": f16(), "pos_palm": f16(3),
         "contact": np.zeros((T, n), np.uint8), "n_at_limit": np.zeros((T, n), np.uint8),
         "joint_speed": f16(), "action_abs": f16(), "reward": f16(len(rnames)), "alive": np.zeros((T, n), np.uint8)}
  if record_actions:
    out["action"] = f16(16)
  if record_quat:
    out["quat_palm"] = f16(4)

  env.reset()
  if assignment is not None:
    assert torch.equal(raw._hand_cube_reset_cache_ids, raw._gr2_assignment)
  obs = env.get_observations()
  cache_ids = raw._hand_cube_reset_cache_ids.clone().cpu().numpy()
  q_prev = cube.data.root_link_quat_w.clone()
  pos0 = cube.data.root_link_pos_w.clone()
  up0 = world_up_in_body(q_prev)
  qj_prev = robot.data.joint_pos.clone()
  first = torch.ones(n, dtype=torch.bool, device=device)
  active = torch.ones(n, dtype=torch.bool, device=device)
  end_step = torch.full((n,), -1, dtype=torch.long, device=device)
  end_code = torch.zeros(n, dtype=torch.int8, device=device)
  init, dr = {}, None
  zero_u = torch.zeros((n, 16), device=device)
  for t in range(T):
    u = actor(obs) if actor is not None else zero_u
    a_t = torch.tanh(u)
    obs, _, dones, _ = env.step(u)
    dones = dones.bool()
    if dr is None:
      dr = {k: v.clone().cpu().numpy() for k, v in GroundTruth(raw).randomization().items()}
    q, pos = cube.data.root_link_quat_w, cube.data.root_link_pos_w
    if bool(first.any()):
      pos0[first] = pos[first]
      up0[first] = world_up_in_body(q[first])
    dpsi = yaw_increment(q, q_prev)
    live = active & ~dones & ~first
    cs = obs["cube_state"]
    cmd = term._target
    lim = ((cmd - lo).abs() < 1e-4) | ((hi - cmd).abs() < 1e-4)
    qj = robot.data.joint_pos
    code = torch.zeros(n, dtype=torch.int8, device=device)
    if bool(dones.any()):
      for name, c in TERM_CODES.items():
        if name in tm.active_terms:
          code = torch.where(tm.get_term(name).bool(), torch.full_like(code, c), code)
    newly = active & dones
    end_step = torch.where(newly, torch.full_like(end_step, t), end_step)
    end_code = torch.where(newly, code, end_code)
    if t == 0:  # initial grasp state as the policy first acts on it (after the first control step)
      palm_c, palm_q = palm_center_pose_w(env=raw, hand_cfg=hand_cfg, palm_center_geom_expr="palm_collision_.*")
      rel = robot.data.geom_pos_w[:, tip_ids] - pos[:, None, :]
      qi = quat_inv(palm_q)[:, None, :].expand(-1, 4, -1).reshape(-1, 4)
      init = {"init_pos_palm": cs[:, 0:3], "init_quat_palm": cs[:, 3:7], "init_contact": cs[:, 13:17] > 0.5,
              "init_tip_rel_palm": quat_apply(qi, rel.reshape(-1, 3)).reshape(n, 4, 3), "init_joint_pos": qj,
              "init_command": cmd, "init_headroom": torch.minimum(cmd - lo, hi - cmd)}
      init = {k: v.clone().float().cpu().numpy() for k, v in init.items()}
    ctc = (cs[:, 13:17] > 0.5).to(torch.uint8)
    rec = {"yaw_inc": torch.where(live, dpsi, 0.0), "drift": (pos - pos0).norm(dim=-1), "tilt": physical_tilt(q, up_ref=up0),
           "pos_palm": cs[:, 0:3], "joint_speed": (qj - qj_prev).abs().mean(-1) / dt, "action_abs": a_t.abs().mean(-1),
           "reward": rm._step_reward * dt}
    if record_actions:
      rec["action"] = a_t
    if record_quat:
      rec["quat_palm"] = cs[:, 3:7]
    for k, v in rec.items():
      out[k][t] = v.float().cpu().numpy()
    out["contact"][t] = (ctc[:, 0] | (ctc[:, 1] << 1) | (ctc[:, 2] << 2) | (ctc[:, 3] << 3)).cpu().numpy()
    out["n_at_limit"][t] = lim.sum(-1).to(torch.uint8).cpu().numpy()
    out["alive"][t] = (active & ~dones).cpu().numpy()
    first = dones.clone()
    q_prev, qj_prev = q.clone(), qj.clone()
    active &= ~dones
  env.close()
  out.update(init)
  out.update({"cache_id": cache_ids, "end_step": end_step.cpu().numpy(), "end_code": end_code.cpu().numpy(),
              "dt": np.float32(dt), "reward_names": np.array(rnames)})
  for k, v in (dr or {}).items():
    out[f"dr_{k}"] = v
  return out


def summary_per_env(tr: dict, steps: int) -> dict[str, np.ndarray]:
  """Per-environment metrics over the first ``steps`` control steps (evaluator definitions)."""
  dt = float(tr["dt"])
  e = tr["end_step"]
  failed = (e >= 0) & (e < steps) & (tr["end_code"] >= 2)
  alive_steps = np.where((e >= 0) & (e < steps), e + 1, steps)
  surv = alive_steps * dt
  rot = tr["yaw_inc"][:steps].astype(np.float64).sum(0)
  return {"turns": rot / (2 * math.pi), "failure": failed.astype(np.float64),
          "pose_failure": (failed & (tr["end_code"] == 4)).astype(np.float64), "survival_s": surv,
          "yaw_rate": rot / np.maximum(surv, dt)}
