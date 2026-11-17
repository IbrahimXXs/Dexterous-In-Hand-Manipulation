#!/usr/bin/env python3
"""Sim rehearsal of the hardware path for a policy run through the LEAP deploy kit.

The robot-side runtime that will drive the real hand (the kit's
``policy_runtime.LeapPolicy``, one instance per simulated environment) runs in
closed loop inside the study environment, and the simulated motors execute
*exactly* its targets (the action term is bypassed), after the same hardware
joint-limit clipping the kit's bridge applies; the clipped command is fed back
to the runtime as on hardware.  The runtime only sees the newest measured
joint-position frame and builds its own history.

Reset semantics at the start of each episode (``--reset``):
  measured : command := measured joint pose (what the kit / repo bridge does)
  target   : command := grasp target, joint history := measured pose (what
             the policy saw in training: fingers rest on the cube short of the
             commanded grasp, and that gap is the grip)

Reported next to the study's reference evaluation (training-stack actor in the
unmodified environment) on the same seed / preset / environments.

    python experiments/scratch_state_estimation/scripts/rehearse_deploy.py \
        --policy ../../LEAP_Hand_RL_Deploy_Kit/policies/A_seed2 --cond A --reset measured --reset target
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[3]
REPO = PROJECT.parents[1]
STUDY = PROJECT / "experiments" / "scratch_state_estimation"
KIT = REPO / "LEAP_Hand_RL_Deploy_Kit" / "kit"
sys.path.insert(0, str(KIT))

from policy_runtime import LeapPolicy, find_policy_files  # noqa: E402

from in_hand_rotation_mjlab.estimation.layout import JOINT_NAMES  # noqa: E402
from in_hand_rotation_mjlab.scratch_study import evaluation as E  # noqa: E402

NEWEST_Q = slice(9 * 16, 10 * 16)
NEWEST_CMD = slice(19 * 16, 20 * 16)
_HW_TO_SERVER = np.array([1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15])
_HW_LO = np.array([-0.523599, -0.261799, -0.261799, -0.261799] * 3 + [0.0, 0.0, -0.349066, -0.349066], np.float32)
_HW_HI = np.array([0.523599, 1.91986, 1.91986, 1.91986] * 3 + [1.571, 1.571, 1.74533, 1.74533], np.float32)


class KitRunner:
  """``evaluate()``-compatible actor; the env executes ``self.targets``."""

  def __init__(self, policy_dir: Path, raw_env, n: int, device: str, reset_mode: str, hw_clip: bool, margin: float):
    self.raw, self.n, self.dev, self.reset_mode, self.hw_clip = raw_env, n, device, reset_mode, hw_clip
    self.pols = [LeapPolicy(policy_dir) for _ in range(n)]
    self.default = self.pols[0].default_joint_pos
    robot = raw_env.scene["robot"]
    term = raw_env.action_manager.get_term("joint_pos")
    names = tuple(robot.joint_names[i] for i in term._target_ids.tolist())
    if names != JOINT_NAMES:
      raise RuntimeError(f"sim joint order {names} != {JOINT_NAMES}")
    if not np.allclose(robot.data.default_joint_pos[0].cpu().numpy(), self.default, atol=1e-6):
      raise RuntimeError("sim default pose differs from the policy's default pose")
    self.lo = _HW_LO[_HW_TO_SERVER] + margin
    self.hi = _HW_HI[_HW_TO_SERVER] - margin
    self.applied = np.zeros((n, 16), np.float32)
    self.targets = None
    self.obs_err_settled: list[float] = []
    self.since_reset = np.zeros(n, np.int64)
    self.clip_events = 0
    self.clip_joint = np.zeros(16)
    self.clip_excess = np.zeros(16)
    self.steps = 0
    term_ref = term

    def process_actions(actions: torch.Tensor):  # motors get exactly the runtime's (clipped) targets
      term_ref._raw_actions[:] = actions
      term_ref._prev_target[:] = term_ref._target
      term_ref._substep_counter = 0
      term_ref._target = self.targets.clone()
      term_ref._processed_actions = term_ref._target
      term_ref.executed_normalized = torch.tanh(actions)

    term.process_actions = process_actions

  @torch.no_grad()
  def __call__(self, obs) -> torch.Tensor:
    ao = obs["actor"].detach().cpu().numpy()
    fresh = (self.raw.episode_length_buf == 0).cpu().numpy()
    raws = np.zeros((self.n, 16), np.float32)
    errs = np.zeros(self.n)
    for i, p in enumerate(self.pols):
      q = ao[i, NEWEST_Q] + self.default
      applied = None
      if fresh[i]:
        p.reset(q)
        if self.reset_mode == "target":
          cmd0 = np.clip(ao[i, NEWEST_CMD], p.lower_limits, p.upper_limits).astype(np.float32)
          p._prev_command = cmd0.copy()
          p._hist["prev_commanded_joint_pos"][:] = cmd0
        self.since_reset[i] = 0
      else:
        applied = self.applied[i]
      target, raw = p.step(q, None, applied)
      out = np.clip(target, self.lo, self.hi) if self.hw_clip else target
      exc = np.abs(out - target)
      self.clip_events += int(np.any(exc > 1e-6))
      self.clip_joint += exc > 1e-6
      self.clip_excess = np.maximum(self.clip_excess, exc)
      self.applied[i] = out
      raws[i] = raw
      errs[i] = np.abs(p.observation() - ao[i]).max()
      self.since_reset[i] += 1
    self.steps += self.n
    settled = self.since_reset > 10
    if settled.any():
      self.obs_err_settled.append(float(errs[settled].max()))
    self.targets = torch.as_tensor(self.applied, device=self.dev)
    return torch.as_tensor(raws, device=self.dev)

  def eval(self):
    return self


KEYS = ("turns", "failure", "drop", "survival_s", "mean_yaw_rate", "exec_saturation")


def run_runner(policy_dir, ckpt, cond, seed, preset, cfg, reset_mode, hw_clip, margin):
  holder = {}
  orig = E.load_study_policy

  def patched(env, agent_cfg, checkpoint, device):
    holder["a"] = KitRunner(policy_dir, env.unwrapped if hasattr(env, "unwrapped") else env, cfg.num_envs,
                            device, reset_mode, hw_clip, margin)
    return holder["a"]

  E.load_study_policy = patched
  try:
    pe, _ = E.evaluate(cond, str(ckpt), seed, preset, cfg)
  finally:
    E.load_study_policy = orig
  a = holder["a"]
  return pe, {
    "obs_max_abs_err_after_10_steps": max(a.obs_err_settled) if a.obs_err_settled else None,
    "hw_limit_clip_fraction": a.clip_events / max(1, a.steps),
    "hw_clip_fraction_per_joint": {n: round(float(c) / max(1, a.steps), 4) for n, c in zip(JOINT_NAMES, a.clip_joint) if c},
    "hw_clip_max_excess_rad": {n: round(float(e), 3) for n, e in zip(JOINT_NAMES, a.clip_excess) if e > 1e-6},
  }


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--policy", type=Path, required=True, help="kit policy folder (model_*.pt + params/)")
  p.add_argument("--cond", default="A", choices=["A", "C"], help="study env variant matching the policy")
  p.add_argument("--reset", action="append", choices=["measured", "target"], default=None)
  p.add_argument("--no-hw-clip", action="store_true")
  p.add_argument("--limit-margin-rad", type=float, default=0.01)
  p.add_argument("--seed", type=int, default=5001)
  p.add_argument("--preset", default="nominal")
  p.add_argument("--num-envs", type=int, default=64)
  p.add_argument("--steps", type=int, default=600)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--out", type=Path, default=STUDY / "results" / "deploy_rehearsal.json")
  a = p.parse_args()
  resets = a.reset or ["measured", "target"]
  ckpt, _, _ = find_policy_files(a.policy)
  cfg = E.EvalConfig(num_envs=a.num_envs, num_steps=a.steps, device=a.device)
  ref, _ = E.evaluate(a.cond, str(ckpt), a.seed, a.preset, cfg)
  res = {"policy": a.policy.name, "checkpoint": ckpt.name, "seed": a.seed, "preset": a.preset,
         "num_envs": a.num_envs, "steps": a.steps, "hw_clip": not a.no_hw_clip,
         "reference_training_actor": {k: float(np.mean(ref[k])) for k in KEYS}}
  for r in resets:
    pe, diag = run_runner(a.policy, ckpt, a.cond, a.seed, a.preset, cfg, r, not a.no_hw_clip, a.limit_margin_rad)
    d = pe["turns"] - ref["turns"]
    res[f"kit_reset_{r}"] = {**{k: float(np.mean(pe[k])) for k in KEYS},
                             "turns_minus_reference": float(d.mean()),
                             "turns_diff_ci95": float(1.96 * d.std(ddof=1) / np.sqrt(len(d))), **diag}
  print(json.dumps(res, indent=2), flush=True)
  allres = json.loads(a.out.read_text()) if a.out.exists() else {}
  allres[f"{a.policy.name}/{a.preset}/seed{a.seed}/n{a.num_envs}"] = res
  a.out.parent.mkdir(parents=True, exist_ok=True)
  a.out.write_text(json.dumps(allres, indent=2) + "\n")


if __name__ == "__main__":
  main()
