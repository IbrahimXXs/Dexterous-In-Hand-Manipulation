#!/usr/bin/env python3
"""Baseline-versus-augmented policy evaluation under identical conditions.

Every checkpoint is evaluated on the same held-out seeds, environment variant
and randomisation preset.  Each environment runs exactly one episode of
``--num-steps`` control steps (the time limit is pushed beyond the window),
using the deterministic policy mean, and is tracked until its first
termination.

Metrics per environment (aggregated to mean / 95% CI in the JSON):
  reward_sum, survival_s, rotation_rad, turns, mean_yaw_rate (rad/s over the
  active time), rotating_fraction (share of 1 s windows with yaw rate >
  0.1 rad/s), drop (cube_fell), failure (any non-timeout termination),
  end_reason, max/mean position drift (mm) and tilt (deg) from the episode
  anchor (first post-reset step), and the randomisation values in effect.

If ``--estimator`` is given (or the policy carries its own estimator), the
estimator is also scored *closed-loop* on the states that policy visits.

Example::

    CUDA_VISIBLE_DEVICES=1 python scripts/estimation/evaluate_policies.py \
        --policy A_model400=../../checkpoint/model_400.pt \
        --policy C_estimator=logs/rsl_rl/.../model_700.pt \
        --seeds 5001 5002 --presets nominal heavy_slippery \
        --output-dir data/eval/run1
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from in_hand_rotation_mjlab.estimation.losses import MetricAccumulator
from in_hand_rotation_mjlab.estimation.model import load_estimator
from in_hand_rotation_mjlab.estimation.policy import EstimatorAugmentedMLPModel
from in_hand_rotation_mjlab.estimation.sim import GroundTruth, load_policy, make_env

# Stress presets widen randomisation beyond the training ranges.
PRESETS: dict[str, dict] = {
  "nominal": {},
  "heavy_slippery": {
    "dr_cube_mass": {"mass_range": (1.3, 1.8)},
    "dr_shared_contact_friction": {"friction_range": (0.4, 0.7)},
  },
  "light_grippy": {
    "dr_cube_mass": {"mass_range": (0.5, 0.8)},
    "dr_shared_contact_friction": {"friction_range": (1.2, 1.6)},
  },
  "high_delay": {"dr_action_delay": {"lag_range": (14, 20)}},
  "weak_motors": {"dr_effort_limits": {"effort_limit": (0.21, 0.26)}},
}


def parse_args():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--policy", action="append", required=True, help="NAME=PATH (repeatable)")
  p.add_argument("--seeds", type=int, nargs="+", default=[5001, 5002])
  p.add_argument("--presets", nargs="+", default=["nominal"], choices=list(PRESETS))
  p.add_argument("--num-envs", type=int, default=2048)
  p.add_argument("--num-steps", type=int, default=600)
  p.add_argument(
    "--env-variant",
    choices=["fixed", "original", "physical_tilt", "no_pose_termination"],
    default="fixed",
    help="'fixed' = corrected cube_pose_deviation reset anchor; 'original' = as shipped; "
    "'physical_tilt' = fixed anchor + physical (Euler-free) tilt angle; "
    "'no_pose_termination' = cube_pose_deviation removed (drops still terminate)",
  )
  p.add_argument("--estimator", default=None, help="score this estimator closed-loop for every policy")
  p.add_argument(
    "--ablation",
    choices=list(ABLATIONS),
    default="none",
    help="estimator-feature ablation for estimator-augmented actors (diagnostics): "
    "zero = features set to 0; shuffle = each env receives the features of env "
    "(i + N/2) mod N, a fixed partner with its own seed / randomisation; "
    "mean = features replaced by their mean over held-out test data",
  )
  p.add_argument(
    "--obs-noise",
    choices=["train", "off"],
    default="train",
    help="actor observation corruption: 'train' (as in training) or 'off' (play mode)",
  )
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--output-dir", required=True)
  return p.parse_args()


ABLATIONS = ("none", "zero", "shuffle", "mean")


def apply_feature_ablation(actor, mode: str, num_envs: int, device: str, data_root: str = "data/estimator") -> None:
  """Replace the estimator features inside an estimator-augmented actor.

  Only the 19/4 features appended to the actor input are changed; the 320-D
  observation path and every weight stay untouched.
  """
  if mode == "none":
    return
  if not isinstance(actor, EstimatorAugmentedMLPModel):
    raise ValueError("--ablation needs an estimator-augmented actor")
  original = actor.estimator_features
  if mode == "mean":
    from in_hand_rotation_mjlab.estimation.dataset import load_split_tensors

    obs = load_split_tensors(data_root, "test", device, max_rows=200_000)["obs"].float()
    with torch.no_grad():
      mean_feat = torch.cat([original(obs[i : i + 50_000]) for i in range(0, obs.shape[0], 50_000)]).mean(0)
  partner = (torch.arange(num_envs, device=device) + num_envs // 2) % num_envs

  def ablated(raw: torch.Tensor) -> torch.Tensor:
    f = original(raw)
    if mode == "zero":
      return torch.zeros_like(f)
    if mode == "shuffle":
      return f[partner]
    return mean_feat.expand_as(f)

  actor.estimator_features = ablated


def ci95(x: np.ndarray) -> float:
  return float(1.96 * x.std(ddof=1) / math.sqrt(max(len(x), 1))) if len(x) > 1 else 0.0


def summarize(per_env: dict[str, np.ndarray]) -> dict:
  out = {}
  for k, v in per_env.items():
    if k.startswith("dr_") or k == "end_reason":
      continue
    v = v.astype(np.float64)
    out[k] = {"mean": float(v.mean()), "ci95": ci95(v), "median": float(np.median(v))}
  er = per_env["end_reason"]
  out["end_reason_counts"] = {str(int(c)): int((er == c).sum()) for c in np.unique(er)}
  return out


@torch.no_grad()
def run_one(name, ckpt, seed, preset, args, estimator) -> tuple[dict, dict, dict | None]:
  dev = args.device
  env, raw, agent_cfg = make_env(
    args.num_envs,
    seed,
    dev,
    episode_length_s=(args.num_steps + 20) * 0.05,
    termination_variant=args.env_variant,
    dr_overrides=PRESETS[preset],
    obs_corruption=None if args.obs_noise == "train" else False,
  )
  gt = GroundTruth(raw)
  _, actor = load_policy(env, agent_cfg, ckpt, dev)
  apply_feature_ablation(actor, args.ablation, args.num_envs, dev)
  own_est = actor.estimator if isinstance(actor, EstimatorAugmentedMLPModel) else None
  scorer = own_est if own_est is not None else estimator
  acc = MetricAccumulator(dev) if scorer is not None else None

  # Fresh, synchronised episode start for every env.
  env.reset()
  obs = env.get_observations()
  n, dt = args.num_envs, float(raw.step_dt)
  cube = raw.scene["cube"]
  from mjlab.utils.lab_api.math import euler_xyz_from_quat, wrap_to_pi

  active = torch.ones(n, dtype=torch.bool, device=dev)
  steps = torch.zeros(n, device=dev)
  reward = torch.zeros(n, device=dev)
  rot = torch.zeros(n, device=dev)
  end_reason = torch.zeros(n, dtype=torch.int8, device=dev)
  drift_max = torch.zeros(n, device=dev)
  drift_sum = torch.zeros(n, device=dev)
  tilt_max = torch.zeros(n, device=dev)
  tilt_sum = torch.zeros(n, device=dev)
  win = int(round(1.0 / dt))
  win_rot = torch.zeros(n, device=dev)
  win_good = torch.zeros(n, device=dev)
  win_count = torch.zeros(n, device=dev)
  dr0 = None
  anchor_pos = anchor_roll = anchor_pitch = prev_yaw = None

  for t in range(args.num_steps):
    action = actor(obs)
    obs, rew, dones, _ = env.step(action)
    dones = dones.bool()
    reasons = gt.end_reasons(dones)
    pos_w, quat_w = gt.world_pose()
    roll, pitch, yaw = euler_xyz_from_quat(quat_w)
    if t == 0:
      anchor_pos, anchor_roll, anchor_pitch = pos_w.clone(), roll.clone(), pitch.clone()
      prev_yaw = yaw.clone()
      dr0 = {k: v.clone() for k, v in gt.randomization().items()}
    live = active & ~dones  # transition stayed inside the same episode
    reward += torch.where(active, rew, 0.0)
    steps += active.float()
    dyaw = wrap_to_pi(yaw - prev_yaw)
    prev_yaw = yaw.clone()
    rot += torch.where(live, dyaw, 0.0)
    win_rot += torch.where(live, dyaw, 0.0)
    drift = (pos_w - anchor_pos).norm(dim=-1)
    tilt = torch.stack(
      [wrap_to_pi(roll - anchor_roll).abs(), wrap_to_pi(pitch - anchor_pitch).abs()], -1
    ).norm(dim=-1)
    drift_max = torch.where(live, torch.maximum(drift_max, drift), drift_max)
    tilt_max = torch.where(live, torch.maximum(tilt_max, tilt), tilt_max)
    drift_sum += torch.where(live, drift, 0.0)
    tilt_sum += torch.where(live, tilt, 0.0)
    if (t + 1) % win == 0:
      full = live
      win_good += (full & (win_rot / (win * dt) > 0.1)).float()
      win_count += full.float()
      win_rot.zero_()
    if acc is not None and live.any():
      labels = gt.labels(obs["critic"])
      pred = scorer.predict(obs["actor"])
      sel = live & (raw.episode_length_buf > 0)
      if sel.any():
        b = {k: v[sel] for k, v in labels.items()}
        b["contact_valid"] = torch.ones(int(sel.sum()), dtype=torch.uint8, device=dev)
        acc.update({k: v[sel] for k, v in pred.items() if v.ndim >= 2}, b)
    end_reason = torch.where(active & dones, reasons, end_reason)
    active &= ~dones
    if not active.any():
      break

  surv = steps * dt
  live_steps = steps.clamp_min(1)
  per_env = {
    "reward_sum": reward,
    "survival_s": surv,
    "rotation_rad": rot,
    "turns": rot / (2 * math.pi),
    "mean_yaw_rate": rot / surv.clamp_min(dt),
    "rotating_fraction": win_good / win_count.clamp_min(1),
    "drop": (end_reason == 2).float(),
    "failure": ((end_reason >= 2)).float(),
    "completed": (end_reason == 0).float(),
    "drift_max_mm": drift_max * 1000,
    "drift_mean_mm": drift_sum / live_steps * 1000,
    "tilt_max_deg": torch.rad2deg(tilt_max),
    "tilt_mean_deg": torch.rad2deg(tilt_sum / live_steps),
    "end_reason": end_reason,
  }
  per_env.update({k: v for k, v in dr0.items()})
  per_env = {k: v.detach().cpu().numpy() for k, v in per_env.items()}
  est_metrics = acc.compute() if acc is not None else None
  env.close()
  return summarize(per_env), per_env, est_metrics


def main():
  args = parse_args()
  out = Path(args.output_dir)
  out.mkdir(parents=True, exist_ok=True)
  estimator = load_estimator(args.estimator, map_location=args.device) if args.estimator else None
  policies = dict(p.split("=", 1) for p in args.policy)
  results: dict = {"config": vars(args), "results": {}}
  for name, ckpt in policies.items():
    results["results"][name] = {"checkpoint": str(Path(ckpt).resolve())}
    for preset in args.presets:
      for seed in args.seeds:
        t0 = time.time()
        summ, per_env, est = run_one(name, ckpt, seed, preset, args, estimator)
        key = f"{preset}/seed_{seed}"
        results["results"][name][key] = {"summary": summ, "estimator_closed_loop": est}
        np.savez_compressed(out / f"{name}__{preset}__seed{seed}.npz", **per_env)
        s = summ
        print(
          f"[{name} | {key}] turns {s['turns']['mean']:.3f}±{s['turns']['ci95']:.3f}  "
          f"yaw_rate {s['mean_yaw_rate']['mean']:.3f}  drop {s['drop']['mean']:.3%}  "
          f"failure {s['failure']['mean']:.3%}  survival {s['survival_s']['mean']:.1f}s  "
          f"reward {s['reward_sum']['mean']:.2f}  ({time.time() - t0:.0f}s)",
          flush=True,
        )
        (out / "results.json").write_text(json.dumps(results, indent=2))
  print(f"[done] {out / 'results.json'}")


if __name__ == "__main__":
  main()
