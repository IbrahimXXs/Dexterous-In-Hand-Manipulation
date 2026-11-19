#!/usr/bin/env python3
"""Pre-training reward audit of Retention-V1 (simulation only).

Rolls out, in Mjlab-Leap-RetV1-A with training-time episode semantics (20-s episodes,
automatic resets) and the reward-weight curriculum frozen at the hwlimit A seed-1 values
reached at the end of its training (the regime a converged policy trains in):
  * the three hwlimit A policies (deterministic), and
  * a "hold" policy (zero action: every joint command held, i.e. no rotation attempt).
For every reward term: mean per-step contribution (weight x raw x dt, as PPO sees it),
separately on steps of rotating vs failing episodes; the share of steps on which the
retention penalty is non-zero; the return per episode; grasp-loss events vs
pose-deviation terminations, falls and timeouts (each event must be counted exactly once
and never on a timeout).

    python experiments/retention_v1/scripts/reward_audit.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

E = Path("experiments/retention_v1")
H = Path("experiments/hwlimit_ac/checkpoints")
POLICIES = {"A1": "A_seed1_selected_model_2600.pt", "A2": "A_seed2_selected_model_2900.pt", "A3": "A_seed3_selected_model_2600.pt", "hold": None}
# hwlimit A seed 1, curriculum weights at update 3000 (TensorBoard Curriculum/*/weight)
FINAL_CURRICULUM = {"object_linvel_penalty": -0.095, "pose_diff_penalty": -0.0317, "torque_penalty": -0.317, "work_penalty": -0.0317}


@torch.no_grad()
def run(task, ck, seed, n, steps, device, episode_length_s=None):
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  from in_hand_rotation_mjlab.estimation.sim import _import_tasks
  from in_hand_rotation_mjlab.hwlimit_study.evaluation import load_study_policy

  _import_tasks()
  cfg = load_env_cfg(task)
  cfg.scene.num_envs, cfg.seed, cfg.curriculum = n, seed, {}
  for k, w in FINAL_CURRICULUM.items():
    cfg.rewards[k].weight = w
  if episode_length_s is not None:
    cfg.episode_length_s = episode_length_s
  agent = load_rl_cfg(task)
  raw = ManagerBasedRlEnv(cfg=cfg, device=device)
  env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
  actor = load_study_policy(env, agent, ck, device) if ck else None
  rm, tm = raw.reward_manager, raw.termination_manager
  names = list(rm._term_names)
  dt = raw.step_dt
  env.reset()
  obs = env.get_observations()
  per_step = np.zeros((steps, n, len(names)), np.float32)
  ev = {"pose": 0, "fell": 0, "timeout": 0, "grasp_loss_events": 0, "grasp_loss_on_timeout": 0, "grasp_loss_on_fell": 0,
        "grasp_loss_without_pose": 0}
  gl = names.index("grasp_loss")
  for t in range(steps):
    u = actor(obs) if actor is not None else torch.zeros(n, 16, device=device)
    obs, rew, dones, extras = env.step(u)
    per_step[t] = (rm._step_reward * dt).cpu().numpy()
    pose, fell = tm.get_term("cube_pose_deviation").bool(), tm.get_term("cube_fell").bool()
    to = tm.time_outs.bool()
    g = rm._step_reward[:, gl] != 0
    ev["pose"] += int((pose & ~fell).sum()); ev["fell"] += int(fell.sum()); ev["timeout"] += int((to & ~pose & ~fell).sum())
    ev["grasp_loss_events"] += int(g.sum()); ev["grasp_loss_on_timeout"] += int((g & to & ~pose).sum())
    ev["grasp_loss_on_fell"] += int((g & fell).sum()); ev["grasp_loss_without_pose"] += int((g & ~pose).sum())
  env.close()
  return names, per_step, ev


def main():
  dev = "cuda:0"
  out = {"curriculum_weights_used": FINAL_CURRICULUM, "policies": {}}
  for name, ck in POLICIES.items():
    names, ps, ev = run("Mjlab-Leap-RetV1-A", str(H / ck) if ck else None, 9301, 512, 800, dev)
    ret = names.index("retention_penalty")
    total = ps.sum(-1)
    out["policies"][name] = {
      "mean_per_step_by_term": dict(zip(names, ps.mean((0, 1)).round(6).tolist())),
      "mean_total_per_step": float(total.mean()),
      "return_per_episode_proxy": float(total.sum(0).mean() / max(1, (ev["pose"] + ev["fell"] + ev["timeout"]) / 512 + 1)),
      "retention_nonzero_step_share": float((ps[..., ret] != 0).mean()),
      "retention_mean_when_nonzero": float(ps[..., ret][ps[..., ret] != 0].mean()) if (ps[..., ret] != 0).any() else 0.0,
      "events": ev,
    }
    print(name, json.dumps(out["policies"][name]), flush=True)
  # timeout check: short episodes, A1
  names, ps, ev = run("Mjlab-Leap-RetV1-A", str(H / POLICIES["A1"]), 9302, 256, 400, dev, episode_length_s=3.0)
  out["timeout_check_3s_episodes"] = ev
  print("timeout check", ev, flush=True)
  (E / "results" / "reward_audit.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
  main()
