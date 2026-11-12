"""Evaluation protocol shared by every policy of the from-scratch study.

Each environment runs ``num_steps`` control steps (30 s at 20 Hz) from a
fresh, synchronised reset with the deterministic policy mean, under a given
seed and randomisation preset.  Two rotation measures are reported:

* uninterrupted (primary): the first episode only, tracked until its first
  termination; rotation is the sum of genuine yaw increments
  (``geometry.yaw_increment``) over transitions that stay inside the episode.
* with resets (secondary): all valid within-episode transitions over the
  whole window, i.e. after a failure the reset episode keeps contributing,
  but the reset jump itself never counts.

Per-env outputs (saved as .npz) include rotation, positive-direction
rotation, survival, failure type, physical tilt / drift, reward,
1-s rotating fraction, executed-action saturation, and the randomisation
values in effect.  If an estimator is given, it is also scored on the
states the policy visits (closed loop).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from in_hand_rotation_mjlab.robots.leap_hand.leap_right_constants import LEAP_ACTUATOR_EFFORT_LIMIT_NM

from .geometry import physical_tilt, world_up_in_body, yaw_increment

_EFF = float(torch.as_tensor(LEAP_ACTUATOR_EFFORT_LIMIT_NM).float().mean()) if hasattr(LEAP_ACTUATOR_EFFORT_LIMIT_NM, "__len__") else float(LEAP_ACTUATOR_EFFORT_LIMIT_NM)

# Randomisation presets.  "nominal" = training distribution.
PRESETS: dict[str, dict] = {
  "nominal": {},
  "heavy_slippery": {"events": {"dr_cube_mass": {"mass_range": (1.3, 1.8)}, "dr_shared_contact_friction": {"friction_range": (0.4, 0.7)}}},
  "light_grippy": {"events": {"dr_cube_mass": {"mass_range": (0.5, 0.8)}, "dr_shared_contact_friction": {"friction_range": (1.2, 1.6)}}},
  "high_delay": {"events": {"dr_action_delay": {"lag_range": (14, 20)}}},
  "weak_motors": {"events": {"dr_effort_limits": {"effort_limit": (0.65 * _EFF, 0.8 * _EFF)}}},
  "obs_noise_x3": {"obs_noise": 0.03},
}

TERM_CODES = {"time_out": 1, "cube_fell": 2, "cube_too_fast": 3, "cube_pose_deviation": 4, "nan": 5}


def make_study_env(condition: str, num_envs: int, seed: int, device: str, *, preset: str = "nominal",
                   episode_length_s: float | None = None, obs_noise_off: bool = False):
  from in_hand_rotation_mjlab.estimation.sim import _import_tasks
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.noise import UniformNoiseCfg

  _import_tasks()
  task = f"Mjlab-Leap-Scratch-{condition}"
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


def load_study_policy(env, agent_cfg, checkpoint: str, device: str):
  from rsl_rl.runners import OnPolicyRunner

  from in_hand_rotation_mjlab.estimation.sim import prepare_agent_cfg

  runner = OnPolicyRunner(env, prepare_agent_cfg(agent_cfg), log_dir=None, device=device)
  runner.load(str(checkpoint), load_cfg={"actor": True, "critic": True}, strict=True, map_location=device)
  actor = runner.alg.get_policy()
  actor.eval()
  return actor


@dataclass
class EvalConfig:
  num_envs: int = 2048
  num_steps: int = 600
  device: str = "cuda:0"


@torch.no_grad()
def evaluate(condition: str, checkpoint: str, seed: int, preset: str, cfg: EvalConfig, *, estimator=None,
             ablation: str | None = None, feature_mean: torch.Tensor | None = None,
             action_sensitivity: bool = False, obs_noise_off: bool = False) -> tuple[dict, dict]:
  from in_hand_rotation_mjlab.estimation.losses import MetricAccumulator

  dev = cfg.device
  env, raw, agent_cfg = make_study_env(condition, cfg.num_envs, seed, dev, preset=preset,
                                       episode_length_s=(cfg.num_steps + 20) * raw_dt(), obs_noise_off=obs_noise_off)
  actor = load_study_policy(env, agent_cfg, checkpoint, dev)
  has_feat = hasattr(actor, "feature_override")
  if ablation:
    if not has_feat:
      raise ValueError("ablation needs a feature-augmented actor")
    actor.feature_override = ablation
    actor.feature_mean = feature_mean
  acc = MetricAccumulator(dev) if estimator is not None else None
  tm = raw.termination_manager
  cube = raw.scene["cube"]
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
  win = int(round(1.0 / dt))
  win_rot, win_good, win_cnt = z(), z(), z()
  end = torch.zeros(n, dtype=torch.int8, device=dev)
  sens = {k: [] for k in ("zero", "mean", "shuffle")} if action_sensitivity else None
  feats = []
  dr = None
  for t in range(cfg.num_steps):
    u = actor(obs)
    if action_sensitivity and has_feat and t % 5 == 0:
      base = torch.tanh(u)
      for mode in sens:
        if mode == "mean" and feature_mean is None:
          continue
        actor.feature_override, actor.feature_mean = mode, feature_mean
        sens[mode].append((torch.tanh(actor(obs)) - base).abs()[active])
        actor.feature_override = ablation
    if has_feat and t % 10 == 0 and actor.last_features is not None:
      feats.append(actor.last_features[active].float().cpu())
    sat_sum += torch.where(active, (torch.tanh(u).abs() > 0.99).float().mean(-1) if agent_cfg.clip_actions is None
                           else (u.abs() >= 1).float().mean(-1), 0.0)
    obs, rew, dones, _ = env.step(u)
    dones = dones.bool()
    if dr is None:
      from in_hand_rotation_mjlab.estimation.sim import GroundTruth

      gt = GroundTruth(raw)
      dr = {k: v.clone() for k, v in gt.randomization().items()}
    q = cube.data.root_link_quat_w
    pos = cube.data.root_link_pos_w
    dpsi = yaw_increment(q, q_prev)
    valid = ~dones & ~first  # transition stays inside one episode
    if bool(first.any()):  # anchor on the first post-reset step
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
      cs = obs["cube_state"]
      pred = estimator(obs["actor"])
      sel = live
      lab = {"pos": cs[sel, 0:3], "quat": cs[sel, 3:7] / cs[sel, 3:7].norm(dim=-1, keepdim=True),
             "lin_vel": cs[sel, 7:10], "ang_vel": cs[sel, 10:13], "contact": cs[sel, 13:17].to(torch.uint8),
             "contact_valid": torch.ones(int(sel.sum()), dtype=torch.uint8, device=dev)}
      acc.update({k: v[sel] for k, v in pred.items() if v.ndim >= 2}, lab)
    if bool(dones.any()):
      code = torch.zeros(n, dtype=torch.int8, device=dev)
      for name, c in TERM_CODES.items():
        if name in tm.active_terms:
          code = torch.where(tm.get_term(name).bool(), torch.full_like(code, c), code)
      end = torch.where(active & dones, code, end)
    first = dones.clone()  # envs that just reset anchor on the next step
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
    "end_reason": end,
  }
  per_env.update(dr or {})
  per_env = {k: v.detach().cpu().numpy() for k, v in per_env.items()}
  extra: dict = {}
  if acc is not None:
    extra["estimator_closed_loop"] = acc.compute()
  if sens is not None:
    extra["executed_action_change"] = {
      k: {"mean_abs": float(torch.cat(v).mean()), "frac_gt_0.05": float((torch.cat(v) > 0.05).float().mean())}
      for k, v in sens.items() if v
    }
  if feats:
    F = torch.cat(feats)
    extra["feature_mean"] = F.mean(0).tolist()
    extra["feature_std"] = F.std(0).tolist()
  env.close()
  return per_env, extra


def raw_dt() -> float:
  return 0.05


def summarize(per_env: dict) -> dict:
  out = {}
  for k, v in per_env.items():
    if k.startswith("dr_") or k == "end_reason":
      continue
    v = np.asarray(v, dtype=np.float64)
    out[k] = {"mean": float(v.mean()), "ci95": float(1.96 * v.std(ddof=1) / math.sqrt(len(v))), "median": float(np.median(v))}
  return out
