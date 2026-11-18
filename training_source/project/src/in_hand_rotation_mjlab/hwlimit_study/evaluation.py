"""Evaluation protocol of the hardware-limit A/C study.

The rollout protocol and every metric shared with the from-scratch study are
the same as ``scratch_study.evaluation.evaluate`` (that module is not
modified): each environment runs ``num_steps`` control steps (30 s at 20 Hz)
from a fresh synchronised reset with the deterministic policy mean; the
primary measure is uninterrupted genuine turns of the first episode.  The
differences are:

* the task id is a parameter (``Mjlab-Leap-HwLim-A/C`` here; the original
  ``Mjlab-Leap-Scratch-*`` tasks can be evaluated for reference);
* deployment-feasibility metrics, measured on the command the motors receive:
  - ``cmd_at_limit_frac``: fraction of joint-steps whose command sits on an
    approved limit (within 1e-4 rad);
  - ``any_joint_at_limit_frac``: fraction of steps with at least one joint
    commanded at an approved limit (the hardware session's
    "steps with a joint at the controller limit" uses the kit's limits);
  - ``kit_clip_frac``: fraction of joint-steps the deployment kit would clip
    (command outside the kit limits incl. margin; 0 by construction in the
    HwLim tasks, the reference value for the original policies);
  - ``max_cmd_step_rad``: largest per-step command change (the kit aborts a
    powered run above 0.05 rad);
  - ``state_outside_frac``: fraction of joint-steps whose measured joint is
    more than 0.02 rad outside the approved limits (pushed by contact);
  - ``action_change``: mean |delta tanh(u)| between consecutive steps;
* optional closed-loop scoring of an estimator on the visited states, plus the
  error of the 19 actor features it produces (``features_from_estimator`` vs
  ``features_from_raw_state`` on the same states) and the input distribution
  shift relative to the estimator's own normaliser.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from in_hand_rotation_mjlab.scratch_study.evaluation import PRESETS, TERM_CODES, EvalConfig, load_study_policy, raw_dt
from in_hand_rotation_mjlab.scratch_study.features import FEATURE_NAMES, features_from_estimator, features_from_raw_state
from in_hand_rotation_mjlab.scratch_study.geometry import physical_tilt, world_up_in_body, yaw_increment

from .limits import kit_limits_sim_order

__all__ = ["PRESETS", "EvalConfig", "make_env", "evaluate", "summarize"]


def make_env(task: str, num_envs: int, seed: int, device: str, *, preset: str = "nominal",
             episode_length_s: float | None = None, obs_noise_off: bool = False):
  from in_hand_rotation_mjlab.estimation.sim import _import_tasks
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.noise import UniformNoiseCfg

  _import_tasks()
  env_cfg = load_env_cfg(task)
  env_cfg.scene.num_envs = int(num_envs)
  env_cfg.seed = int(seed)
  env_cfg.curriculum = {}  # penalties at their configured (final) weights
  if episode_length_s is not None:
    env_cfg.episode_length_s = float(episode_length_s)
  p = PRESETS[preset]
  for name, params in p.get("events", {}).items():
    env_cfg.events[name].params.update(params)
  if "obs_noise" in p:
    n = p["obs_noise"]
    env_cfg.observations["actor"].terms["joint_pos"].noise = UniformNoiseCfg(n_min=-n, n_max=n)
  if obs_noise_off:
    env_cfg.observations["actor"].enable_corruption = False
  agent_cfg = load_rl_cfg(task)
  raw = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
  return env, raw, agent_cfg


def _command_limits(raw) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """(approved lo, hi) used by the task's action, and the kit's (lo, hi) incl. margin."""
  term = raw.action_manager.get_term("joint_pos")
  dev = raw.device
  if hasattr(term, "approved_limits"):
    lo, hi = term.approved_limits()
  else:  # original task: report against the approved limits for reference
    from .limits import approved_command_limits

    rob = raw.scene["robot"]
    soft = rob.data.soft_joint_pos_limits[0].cpu().numpy()
    lo_np, hi_np = approved_command_limits(soft[:, 0], soft[:, 1])
    lo, hi = torch.as_tensor(lo_np, device=dev, dtype=torch.float32), torch.as_tensor(hi_np, device=dev, dtype=torch.float32)
  klo, khi = kit_limits_sim_order()
  return lo, hi, torch.as_tensor(klo, device=dev, dtype=torch.float32), torch.as_tensor(khi, device=dev, dtype=torch.float32)


@torch.no_grad()
def evaluate(task: str, checkpoint: str, seed: int, preset: str, cfg: EvalConfig, *, estimator=None,
             obs_noise_off: bool = False) -> tuple[dict, dict]:
  from in_hand_rotation_mjlab.estimation.losses import MetricAccumulator

  dev = cfg.device
  env, raw, agent_cfg = make_env(task, cfg.num_envs, seed, dev, preset=preset,
                                 episode_length_s=(cfg.num_steps + 20) * raw_dt(), obs_noise_off=obs_noise_off)
  actor = load_study_policy(env, agent_cfg, checkpoint, dev)
  has_feat = hasattr(actor, "feature_override")
  acc = MetricAccumulator(dev) if estimator is not None else None
  tm = raw.termination_manager
  cube = raw.scene["cube"]
  robot = raw.scene["robot"]
  term = raw.action_manager.get_term("joint_pos")
  lo, hi, klo, khi = _command_limits(raw)
  n, dt = cfg.num_envs, float(raw.step_dt)

  env.reset()
  obs = env.get_observations()
  q_prev = cube.data.root_link_quat_w.clone()
  pos0 = cube.data.root_link_pos_w.clone()
  up0 = world_up_in_body(q_prev)
  first = torch.ones(n, dtype=torch.bool, device=dev)
  active = torch.ones(n, dtype=torch.bool, device=dev)
  z = lambda: torch.zeros(n, device=dev)  # noqa: E731
  steps, reward, rot, rot_pos, rot_all = z(), z(), z(), z(), z()
  drift_max, drift_sum, tilt_max, tilt_sum, sat_sum = z(), z(), z(), z(), z()
  at_lim, any_lim, kit_clip, outside, act_chg, max_step = z(), z(), z(), z(), z(), z()
  joint_at_lim = torch.zeros(16, device=dev)
  win = int(round(1.0 / dt))
  win_rot, win_good, win_cnt = z(), z(), z()
  end = torch.zeros(n, dtype=torch.int8, device=dev)
  feats, dr = [], None
  feat_se, feat_n = torch.zeros(len(FEATURE_NAMES), device=dev, dtype=torch.float64), 0
  zs_sum, zs_gt4, zs_n = 0.0, 0.0, 0
  prev_a = None
  prev_cmd = term._target.clone()
  for t in range(cfg.num_steps):
    u = actor(obs)
    a_t = torch.tanh(u) if agent_cfg.clip_actions is None else u.clamp(-1, 1)
    if has_feat and t % 10 == 0 and actor.last_features is not None:
      feats.append(actor.last_features[active].float().cpu())
    sat_sum += torch.where(active, (a_t.abs() > 0.99).float().mean(-1), 0.0)
    if prev_a is not None:
      act_chg += torch.where(active & ~first, (a_t - prev_a).abs().mean(-1), 0.0)
    prev_a = a_t
    obs, rew, dones, _ = env.step(u)
    dones = dones.bool()
    cmd = term._target  # the command the motors track this step (post-reset for done envs)
    lim = ((cmd - lo).abs() < 1e-4) | ((hi - cmd).abs() < 1e-4)
    live_cmd = active & ~dones
    at_lim += torch.where(live_cmd, lim.float().mean(-1), 0.0)
    any_lim += torch.where(live_cmd, lim.any(-1).float(), 0.0)
    joint_at_lim += (lim & live_cmd[:, None]).float().sum(0)
    kit_clip += torch.where(live_cmd, ((cmd < klo - 1e-6) | (cmd > khi + 1e-6)).float().mean(-1), 0.0)
    step_now = (cmd - prev_cmd).abs().amax(-1)
    max_step = torch.where(live_cmd & ~first, torch.maximum(max_step, step_now), max_step)
    q_meas = robot.data.joint_pos
    outside += torch.where(live_cmd, ((q_meas < lo - 0.02) | (q_meas > hi + 0.02)).float().mean(-1), 0.0)
    prev_cmd = cmd.clone()
    if dr is None:
      from in_hand_rotation_mjlab.estimation.sim import GroundTruth

      dr = {k: v.clone() for k, v in GroundTruth(raw).randomization().items()}
    q = cube.data.root_link_quat_w
    pos = cube.data.root_link_pos_w
    dpsi = yaw_increment(q, q_prev)
    valid = ~dones & ~first
    if bool(first.any()):
      pos0[first] = pos[first]
      up0[first] = world_up_in_body(q[first])
    live = active & valid
    reward += torch.where(active, rew, 0.0)
    steps += active.float()
    rot += torch.where(live, dpsi, 0.0)
    rot_pos += torch.where(live, dpsi.clamp_min(0.0), 0.0)
    rot_all += torch.where(valid, dpsi, 0.0)
    win_rot += torch.where(live, dpsi, 0.0)
    drift = (pos - pos0).norm(dim=-1)
    tilt = physical_tilt(q, up_ref=up0)
    drift_max = torch.where(live, torch.maximum(drift_max, drift), drift_max)
    tilt_max = torch.where(live, torch.maximum(tilt_max, tilt), tilt_max)
    drift_sum += torch.where(live, drift, 0.0)
    tilt_sum += torch.where(live, tilt, 0.0)
    if (t + 1) % win == 0:
      win_good += (live & (win_rot / (win * dt) > 0.1)).float()
      win_cnt += live.float()
      win_rot.zero_()
    if acc is not None and bool(live.any()):
      cs = obs["cube_state"][live]
      xo = obs["actor"][live]
      pred = estimator(xo)
      lab = {"pos": cs[:, 0:3], "quat": cs[:, 3:7] / cs[:, 3:7].norm(dim=-1, keepdim=True),
             "lin_vel": cs[:, 7:10], "ang_vel": cs[:, 10:13], "contact": cs[:, 13:17].to(torch.uint8),
             "contact_valid": torch.ones(cs.shape[0], dtype=torch.uint8, device=dev)}
      acc.update({k: v for k, v in pred.items() if v.ndim >= 2}, lab)
      fe = features_from_estimator(estimator, xo)
      fg = features_from_raw_state(cs)
      feat_se += ((fe - fg) ** 2).sum(0).double()
      feat_n += cs.shape[0]
      zs = ((xo - estimator.obs_mean) / estimator.obs_std).abs()
      zs_sum += float(zs.mean(-1).sum())
      zs_gt4 += float((zs > 4).float().mean(-1).sum())
      zs_n += cs.shape[0]
    if bool(dones.any()):
      code = torch.zeros(n, dtype=torch.int8, device=dev)
      for name, c in TERM_CODES.items():
        if name in tm.active_terms:
          code = torch.where(tm.get_term(name).bool(), torch.full_like(code, c), code)
      end = torch.where(active & dones, code, end)
    first = dones.clone()
    q_prev = q.clone()
    active &= ~dones

  surv = steps * dt
  live_steps = steps.clamp_min(1)
  per_env = {
    "turns": rot / (2 * math.pi),
    "rotation_rad": rot,
    "positive_rotation_rad": rot_pos,
    "turns_with_resets": rot_all / (2 * math.pi),
    "mean_yaw_rate": rot / surv.clamp_min(dt),
    "survival_s": surv,
    "completed": (end == 0).float(),
    "drop": (end == 2).float(),
    "pose_failure": (end == 4).float(),
    "too_fast": (end == 3).float(),
    "failure": (end >= 2).float(),
    "reward_sum": reward,
    "rotating_fraction": win_good / win_cnt.clamp_min(1),
    "drift_max_mm": drift_max * 1000,
    "drift_mean_mm": drift_sum / live_steps * 1000,
    "tilt_max_deg": torch.rad2deg(tilt_max),
    "tilt_mean_deg": torch.rad2deg(tilt_sum / live_steps),
    "exec_saturation": sat_sum / live_steps,
    "cmd_at_limit_frac": at_lim / live_steps,
    "any_joint_at_limit_frac": any_lim / live_steps,
    "kit_clip_frac": kit_clip / live_steps,
    "max_cmd_step_rad": max_step,
    "state_outside_frac": outside / live_steps,
    "action_change": act_chg / live_steps,
    "end_reason": end,
  }
  per_env.update(dr or {})
  per_env = {k: v.detach().cpu().numpy() for k, v in per_env.items()}
  from in_hand_rotation_mjlab.estimation.layout import JOINT_NAMES

  extra: dict = {"cmd_at_limit_frac_per_joint": dict(zip(JOINT_NAMES, (joint_at_lim / steps.sum().clamp_min(1)).tolist()))}
  if acc is not None:
    extra["estimator_closed_loop"] = acc.compute()
    extra["feature_rmse"] = dict(zip(FEATURE_NAMES, torch.sqrt(feat_se / max(1, feat_n)).tolist()))
    extra["input_shift"] = {"mean_abs_z": zs_sum / max(1, zs_n), "frac_abs_z_gt4": zs_gt4 / max(1, zs_n), "rows": zs_n}
  if feats:
    F = torch.cat(feats)
    extra["feature_mean"] = F.mean(0).tolist()
    extra["feature_std"] = F.std(0).tolist()
  env.close()
  return per_env, extra


def summarize(per_env: dict) -> dict:
  out = {}
  for k, v in per_env.items():
    if k.startswith("dr_") or k == "end_reason":
      continue
    v = np.asarray(v, dtype=np.float64)
    out[k] = {"mean": float(v.mean()), "ci95": float(1.96 * v.std(ddof=1) / math.sqrt(len(v))), "median": float(np.median(v))}
  return out
