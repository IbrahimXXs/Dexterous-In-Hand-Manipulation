#!/usr/bin/env python3
"""How much do the estimator features actually influence the actor?

Runs the estimator-augmented policy closed-loop (fixed-reset task, held-out
seed), stores the visited raw 320-D observations, and then measures on those
states -- without relying on raw weight magnitudes:

1. Feature scale: mean / std / range of each of the 19 estimator features
   versus the running-normalised 320-D observation the actor also receives.
2. First-layer pre-activation share: variance of W_f f versus W_o z across
   visited states (how much of the hidden-unit signal comes from features).
3. Action saturation: the share of action entries outside the +-1 clip, and
   the share where PPO's exploration noise cannot change the executed action.
4. Counterfactual action changes on identical states, on the *executed*
   (clipped) action: features zeroed / shuffled between envs / replaced by
   their mean, compared with reference perturbations (one extra draw of the
   +-0.01 rad training observation noise; swapping in the A' or model_400
   actor).
5. Gradient attribution: |d a_j / d x_k| * std(x_k), counted only where
   a_j is inside the clip range (elsewhere the executed action does not
   depend on x), summed per input group.

    CUDA_VISIBLE_DEVICES=3 python scripts/estimation/diagnose_feature_influence.py \
        --checkpoint logs/rsl_rl/.../model_699.pt --reference-a-prime ... \
        --reference-model400 ../../checkpoint/model_400.pt --output-dir data/eval/diagnostics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from in_hand_rotation_mjlab.estimation.layout import ACTOR_OBS_DIM  # noqa: E402
from in_hand_rotation_mjlab.estimation.policy import EstimatorAugmentedMLPModel  # noqa: E402
from in_hand_rotation_mjlab.estimation.sim import load_policy, make_env  # noqa: E402

FEATURE_NAMES = {
  "full": [
    "pos_x", "pos_y", "pos_z",
    "rot_c1x", "rot_c1y", "rot_c1z", "rot_c2x", "rot_c2y", "rot_c2z",
    "linvel_x", "linvel_y", "linvel_z",
    "angvel_x", "angvel_y", "angvel_z",
    "contact_if", "contact_mf", "contact_rf", "contact_th",
  ],
  "contact": ["contact_if", "contact_mf", "contact_rf", "contact_th"],
}
FEATURE_GROUPS = {
  "full": {"est_pos": (0, 3), "est_rot6d": (3, 9), "est_lin_vel": (9, 12), "est_ang_vel": (12, 15), "est_contact": (15, 19)},
  "contact": {"est_contact": (0, 4)},
}


def parse_args():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--checkpoint", required=True, help="estimator-augmented policy")
  p.add_argument("--reference-a-prime", default=None, help="plain-MLP control fine-tune (A')")
  p.add_argument("--reference-model400", default=None)
  p.add_argument("--num-envs", type=int, default=1024)
  p.add_argument("--num-steps", type=int, default=400)
  p.add_argument("--sample-every", type=int, default=4)
  p.add_argument("--warmup-steps", type=int, default=20)
  p.add_argument("--seed", type=int, default=5001)
  p.add_argument("--jacobian-states", type=int, default=8192)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--output-dir", required=True)
  return p.parse_args()


def exec_change(a: torch.Tensor, b: torch.Tensor) -> dict:
  d = (a.clamp(-1, 1) - b.clamp(-1, 1)).abs()
  return {
    "executed_mean_abs": float(d.mean()),
    "executed_frac_entries_gt_0.01": float((d > 0.01).float().mean()),
    "executed_frac_entries_gt_0.05": float((d > 0.05).float().mean()),
    "executed_mean_l2_per_state": float(d.norm(dim=-1).mean()),
    "raw_mean_abs": float((a - b).abs().mean()),
  }


@torch.no_grad()
def collect(env, raw_env, actor, args) -> torch.Tensor:
  env.reset()
  obs = env.get_observations()
  keep = []
  for t in range(args.num_steps):
    obs, _, _, _ = env.step(actor(obs))
    if t >= args.warmup_steps and t % args.sample_every == 0:
      ok = raw_env.episode_length_buf > args.warmup_steps // 2  # skip just-reset envs
      keep.append(obs["actor"][ok].clone())
  return torch.cat(keep)


def main():
  args = parse_args()
  out = Path(args.output_dir)
  out.mkdir(parents=True, exist_ok=True)
  dev = args.device
  env, raw_env, agent_cfg = make_env(args.num_envs, args.seed, dev, fix_reset_anchor=True)
  _, actor = load_policy(env, agent_cfg, args.checkpoint, dev)
  if not isinstance(actor, EstimatorAugmentedMLPModel):
    raise SystemExit("--checkpoint must be an estimator-augmented policy")
  mode = actor.feature_mode
  names, groups = FEATURE_NAMES[mode], FEATURE_GROUPS[mode]
  X = collect(env, raw_env, actor, args)  # raw [S, 320]
  S = X.shape[0]
  report: dict = {"checkpoint": str(Path(args.checkpoint).resolve()), "feature_mode": mode, "num_states": S, "seed": args.seed}

  with torch.no_grad():
    z = torch.cat([actor.obs_normalizer(X[i : i + 65536]) for i in range(0, S, 65536)])
    f = torch.cat([actor.estimator_features(X[i : i + 65536]) for i in range(0, S, 65536)])
    lat = torch.cat([z, f], -1)
    a = torch.cat([actor.mlp(lat[i : i + 65536]) for i in range(0, S, 65536)])

    # 1. scales
    report["feature_scale"] = {
      n: {"mean": float(f[:, i].mean()), "std": float(f[:, i].std()), "min": float(f[:, i].min()), "max": float(f[:, i].max())}
      for i, n in enumerate(names)
    }
    zs = z.std(0)
    report["normalised_obs_scale"] = {
      "mean_abs_mean": float(z.mean(0).abs().mean()),
      "std_mean": float(zs.mean()),
      "std_min": float(zs.min()),
      "std_max": float(zs.max()),
    }

    # 2. first-layer pre-activation share
    W = actor.mlp[0].weight
    Wo, Wf = W[:, :ACTOR_OBS_DIM], W[:, ACTOR_OBS_DIM:]
    ho, hf = z @ Wo.T, f @ Wf.T
    vo, vf = ho.var(0), hf.var(0)
    report["first_layer"] = {
      "obs_col_norm_mean": float(Wo.norm(dim=0).mean()),
      "feature_col_norm_mean": float(Wf.norm(dim=0).mean()),
      "var_share_features_total": float(vf.sum() / (vf.sum() + vo.sum())),
      "std_ratio_feature_to_obs_median_unit": float((vf.sqrt() / vo.sqrt().clamp_min(1e-9)).median()),
      "std_ratio_feature_to_obs_max_unit": float((vf.sqrt() / vo.sqrt().clamp_min(1e-9)).max()),
    }

    # 3. saturation and exploration
    sigma = actor.std.detach() if hasattr(actor, "std") else None
    sat = a.abs() > 1
    rep = {"frac_entries_outside_clip": float(sat.float().mean()), "per_joint_frac_outside_clip": sat.float().mean(0).tolist(),
           "raw_action_abs_median": float(a.abs().median())}
    if sigma is not None:
      rep["policy_noise_std_mean"] = float(sigma.mean())
      rep["frac_entries_beyond_clip_plus_3sigma"] = float((a.abs() > 1 + 3 * sigma).float().mean())
      g = torch.Generator(device=dev).manual_seed(0)
      same = 0.0
      for _ in range(8):
        smp = a + sigma * torch.randn(a.shape, device=dev, generator=g)
        same += float(((smp.clamp(-1, 1) - a.clamp(-1, 1)).abs() < 1e-6).float().mean())
      rep["frac_samples_where_exploration_leaves_executed_action_unchanged"] = same / 8
    report["saturation"] = rep

    # 4. counterfactual executed-action changes
    cf = {}
    partner = torch.roll(torch.arange(S, device=dev), S // 2)
    mlp = lambda latent: torch.cat([actor.mlp(latent[i : i + 65536]) for i in range(0, S, 65536)])  # noqa: E731
    cf["features_zeroed"] = exec_change(mlp(torch.cat([z, torch.zeros_like(f)], -1)), a)
    cf["features_shuffled_between_states"] = exec_change(mlp(torch.cat([z, f[partner]], -1)), a)
    cf["features_replaced_by_mean"] = exec_change(mlp(torch.cat([z, f.mean(0, keepdim=True).expand_as(f)], -1)), a)
    noisy = X.clone()
    noisy[:, :160] += (torch.rand_like(noisy[:, :160]) * 2 - 1) * 0.01
    zn = torch.cat([actor.obs_normalizer(noisy[i : i + 65536]) for i in range(0, S, 65536)])
    cf["reference_extra_obs_noise_draw_features_fixed"] = exec_change(mlp(torch.cat([zn, f], -1)), a)
    for label, ck in (("reference_swap_to_A_prime_actor", args.reference_a_prime), ("reference_swap_to_model400_actor", args.reference_model400)):
      if ck:
        _, other = load_policy(env, agent_cfg, ck, dev)
        ao = torch.cat([other.mlp(other.obs_normalizer(X[i : i + 65536])) for i in range(0, S, 65536)])
        cf[label] = exec_change(ao, a)
    report["counterfactual_executed_action_change"] = cf

  # 5. gradient attribution on a subset
  idx = torch.randperm(S, generator=torch.Generator().manual_seed(1))[: args.jacobian_states].to(dev)
  L = lat[idx].clone().requires_grad_(True)
  A = actor.mlp(L)
  live = (A.abs() < 1).float().detach()
  lat_std = lat.std(0)
  centred = (lat[idx] - lat.mean(0)).abs()
  sens_live = torch.zeros(L.shape[1], device=dev)
  sens_raw = torch.zeros(L.shape[1], device=dev)
  gxi_live = torch.zeros(L.shape[1], device=dev)
  for j in range(A.shape[1]):
    (g,) = torch.autograd.grad(A[:, j].sum(), L, retain_graph=True)
    g = g.abs()
    sens_live += (g * live[:, j : j + 1]).mean(0) * lat_std
    sens_raw += g.mean(0) * lat_std
    gxi_live += (g * centred * live[:, j : j + 1]).mean(0)
  all_groups = {"obs_joint_pos_history": (0, 160), "obs_prev_commands": (160, 320)}
  all_groups.update({k: (ACTOR_OBS_DIM + a0, ACTOR_OBS_DIM + a1) for k, (a0, a1) in groups.items()})
  attr = {}
  for name, vec in (("grad_x_std_executed", sens_live), ("grad_x_std_raw_output", sens_raw), ("grad_x_deviation_executed", gxi_live)):
    tot = float(vec.sum())
    attr[name] = {
      g: {"share": float(vec[a0:a1].sum()) / tot, "per_dim_mean": float(vec[a0:a1].mean())} for g, (a0, a1) in all_groups.items()
    }
    attr[name]["estimator_total_share"] = float(vec[ACTOR_OBS_DIM:].sum()) / tot
  attr["frac_entries_inside_clip_on_subset"] = float(live.mean())
  attr["per_feature_grad_x_std_executed"] = {n: float(sens_live[ACTOR_OBS_DIM + i]) for i, n in enumerate(names)}
  report["gradient_attribution"] = attr

  (out / "feature_influence.json").write_text(json.dumps(report, indent=2))

  fig, axes = plt.subplots(1, 2, figsize=(13, 4))
  gnames = list(all_groups)
  for k, (key, title) in enumerate((("grad_x_std_executed", "share of |da/dx|*std(x), executed action"), ("grad_x_std_raw_output", "same, raw (pre-clip) output"))):
    axes[k].bar(range(len(gnames)), [attr[key][g]["share"] for g in gnames], color=["C0", "C0"] + ["C3"] * (len(gnames) - 2))
    axes[k].set_xticks(range(len(gnames)), gnames, rotation=30, fontsize=8)
    axes[k].set_title(title, fontsize=9)
  fig.tight_layout()
  fig.savefig(out / "feature_influence.png", dpi=110)
  plt.close(fig)
  print(json.dumps(report, indent=1))
  env.close()


if __name__ == "__main__":
  main()
