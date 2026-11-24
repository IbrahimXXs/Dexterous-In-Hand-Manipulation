#!/usr/bin/env python3
"""Calibrate the C5-HW-effort penalty weight w ONCE, before training (SPEC.md Section 3).

Rule (pre-registered): w is set so that the mean penalty w * p equals 10% of C5's mean
rotation reward per step, on frozen C5 rollouts with the capped (350 mA-equivalent) actuator.

Rollouts: the hardware C5 checkpoint (C5_1 selected, model 2400, sha checked), the C5-HW-effort
environment (capped effort limits) with the penalty at weight 1 (so its logged value is p),
grasp 49 (the hardware placement, as in results/sim_effort_diagnostic.json), seed 8301,
256 environments, 600 steps (30 s), penalty weights at the frozen final curriculum values
(rollout.make_env).  Means are over the steps each environment is alive in its first episode
(the terminal step included).  Both terms are compared as func * weight (the reward manager
multiplies every term by the same dt).

    python experiments/c5_hw_v1/scripts/calibrate_effort_weight.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[3]
E = PROJECT / "experiments" / "c5_hw_v1"
C5 = ("experiments/grasp_rotation_v2/checkpoints/C5_1_selected_model_2400.pt",
      "2a932dfe37a5540d74a07081a8de62658b56341eca2efcb5af38a9790ef58492")
GRASP, SEED, N, T, SHARE = 49, 8301, 256, 600, 0.10


@torch.no_grad()
def main() -> None:
  import mjlab.tasks.registry as reg
  from in_hand_rotation_mjlab.grasp_rotation_v2 import rollout as R
  from in_hand_rotation_mjlab.hwlimit_study.evaluation import load_study_policy
  from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_c5_hw_v1.env_cfgs import c5hw_effort_env_cfg

  ck = PROJECT / C5[0]
  assert hashlib.sha256(ck.read_bytes()).hexdigest() == C5[1]
  base = reg.load_env_cfg
  reg.load_env_cfg = lambda task, *a, **k: c5hw_effort_env_cfg(weight=1.0)
  try:
    env, raw, agent_cfg = R.make_env("Mjlab-Leap-GRV2-C5", N, SEED, "cuda:0", num_steps=T)
  finally:
    reg.load_env_cfg = base
  raw._gr2_assignment = torch.full((N,), GRASP, dtype=torch.long, device="cuda:0")
  actor = load_study_policy(env, agent_cfg, str(ck), "cuda:0")
  rm = raw.reward_manager
  names = list(rm._term_names)
  ir, ip = names.index("rotate_finite_diff"), names.index("actuator_utilisation_penalty")
  assert rm._term_cfgs[ip].weight == -1.0
  fr = raw.sim.model.actuator_forcerange[:, raw.scene["robot"].indexing.ctrl_ids, 1]
  env.reset()
  obs = env.get_observations()
  lim = fr.clone().cpu().numpy()
  active = torch.ones(N, dtype=torch.bool, device="cuda:0")
  rot, pen, cnt = 0.0, 0.0, 0
  for _ in range(T):
    obs, _, dones, _ = env.step(actor(obs))
    m = active
    rot += float(rm._step_reward[m, ir].double().sum())
    pen += float(-rm._step_reward[m, ip].double().sum())
    cnt += int(m.sum())
    active = active & ~dones.bool()
  env.close()
  rot_mean, p_mean = rot / cnt, pen / cnt
  w = SHARE * rot_mean / p_mean
  out = {"rule": "w * mean(p) = 0.10 * mean(rotation reward per step), frozen C5 under the capped actuator",
         "checkpoint": C5[0], "sha256": C5[1], "task_env": "c5hw_effort_env_cfg(weight=1)", "grasp": GRASP,
         "seed": SEED, "num_envs": N, "steps": T, "alive_env_steps": cnt,
         "effort_limit_range_nm": [float(lim.min()), float(lim.max())],
         "rotation_reward_per_step_mean": rot_mean, "rotation_reward_weight": float(rm._term_cfgs[ir].weight),
         "penalty_p_per_step_mean": p_mean, "share": SHARE, "w": w, "w_rounded_frozen": float(f"{w:.4g}"),
         "cross_check_sim_effort_diagnostic_hinge": 0.228}
  (E / "results").mkdir(exist_ok=True)
  (E / "results" / "effort_weight_calibration.json").write_text(json.dumps(out, indent=1) + "\n")
  print(json.dumps(out, indent=1))


if __name__ == "__main__":
  main()
