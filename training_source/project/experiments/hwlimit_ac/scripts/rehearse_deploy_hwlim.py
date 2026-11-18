#!/usr/bin/env python3
"""Phase 6: simulation-only rehearsal of the deploy kit with the new A/C policies.

Uses the hardware session's ``KitRunner`` (``scratch_state_estimation/scripts/rehearse_deploy.py``,
imported unchanged): one kit ``LeapPolicy`` per simulated environment runs in closed loop,
the bridge's hardware-limit clip (kit limits - 0.01 rad) is applied, the clipped command
is fed back, and the simulated motors execute exactly those commands.  Here the
environment is ``Mjlab-Leap-HwLim-<cond>`` (approved limits, new reset cache).

For each deployment folder: reference = the training actor in the same environment, seed,
preset and environments; kit runtime with reset "measured" (as on the hand) and "target"
(as in training).  Reported: turns and failures, the paired difference with its 95% CI,
how often the bridge clip changes a command (expected ~0), and observation agreement.
No hardware is accessed.

    python experiments/hwlimit_ac/scripts/rehearse_deploy_hwlim.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
E = PROJECT / "experiments" / "hwlimit_ac"
_spec = importlib.util.spec_from_file_location(
  "rehearse_deploy", PROJECT / "experiments" / "scratch_state_estimation" / "scripts" / "rehearse_deploy.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

from in_hand_rotation_mjlab.hwlimit_study import evaluation as H  # noqa: E402

KEYS = ("turns", "failure", "drop", "survival_s", "mean_yaw_rate", "any_joint_at_limit_frac", "max_cmd_step_rad")


def run_kit(policy_dir, ckpt, task, seed, preset, cfg, reset_mode, margin, training_limits=False):
  """``training_limits``: diagnostic only -- the kit runtime's own joint limits (the model's
  hard ranges) are replaced in memory by the limits the training action used (approved
  limits incl. the model's soft limits).  The kit files are not changed."""
  holder = {}
  orig = H.load_study_policy

  def patched(env, agent_cfg, checkpoint, device):
    raw = env.unwrapped if hasattr(env, "unwrapped") else env
    holder["a"] = R.KitRunner(policy_dir, raw, cfg.num_envs, device, reset_mode, True, margin)
    if training_limits:
      lo, hi = (x.cpu().numpy().astype(np.float32) for x in raw.action_manager.get_term("joint_pos").approved_limits())
      for pol in holder["a"].pols:
        pol.lower_limits, pol.upper_limits = lo, hi
    return holder["a"]

  H.load_study_policy = patched
  try:
    pe, _ = H.evaluate(task, str(ckpt), seed, preset, cfg)
  finally:
    H.load_study_policy = orig
  a = holder["a"]
  return pe, {
    "obs_max_abs_err_after_10_steps": max(a.obs_err_settled) if a.obs_err_settled else None,
    "bridge_clip_fraction_of_steps": a.clip_events / max(1, a.steps),
    "bridge_clip_fraction_per_joint": {n: round(float(c) / max(1, a.steps), 5) for n, c in zip(R.JOINT_NAMES, a.clip_joint) if c},
    "bridge_clip_max_excess_rad": {n: round(float(e), 4) for n, e in zip(R.JOINT_NAMES, a.clip_excess) if e > 1e-6},
  }


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--seed", type=int, default=5001)
  p.add_argument("--preset", default="nominal")
  p.add_argument("--num-envs", type=int, default=128)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--out", type=Path, default=E / "results" / "deploy_rehearsal.json")
  p.add_argument("--diagnose", nargs="*", default=None, help="policy folders to rerun with training limits in the kit runtime")
  a = p.parse_args()
  cfg = H.EvalConfig(num_envs=a.num_envs, num_steps=600, device=a.device)
  allres = json.loads(a.out.read_text()) if a.out.exists() else {}
  if a.diagnose is not None:
    for name in a.diagnose:
      d = E / "deploy" / "policies" / name
      key = f"{name}/{a.preset}/seed{a.seed}/n{a.num_envs}"
      ckpt = next(d.glob("model_*.pt"))
      task = f"Mjlab-Leap-HwLim-{name[0]}"
      ref, _ = H.evaluate(task, str(ckpt), a.seed, a.preset, cfg)
      pe, diag = run_kit(d, ckpt, task, a.seed, a.preset, cfg, "target", 0.01, training_limits=True)
      dt = pe["turns"] - ref["turns"]
      allres.setdefault(key, {})["diagnostic_kit_reset_target_training_limits"] = {
        "reference_turns_this_invocation": float(np.mean(ref["turns"])), "turns": float(np.mean(pe["turns"])),
        "turns_minus_reference": float(dt.mean()), "turns_diff_ci95": float(1.96 * dt.std(ddof=1) / np.sqrt(len(dt))), **diag}
      a.out.write_text(json.dumps(allres, indent=1) + "\n")
      print(f"[diagnose {key}] ref {np.mean(ref['turns']):.3f} kit+training limits {np.mean(pe['turns']):.3f} "
            f"({dt.mean():+.3f}±{1.96 * dt.std(ddof=1) / np.sqrt(len(dt)):.3f})", flush=True)
    return
  for d in sorted((E / "deploy" / "policies").iterdir()):
    key = f"{d.name}/{a.preset}/seed{a.seed}/n{a.num_envs}"
    if key in allres:
      continue
    ckpt = next(d.glob("model_*.pt"))
    task = f"Mjlab-Leap-HwLim-{d.name[0]}"
    ref, _ = H.evaluate(task, str(ckpt), a.seed, a.preset, cfg)
    res = {"checkpoint": ckpt.name, "reference_training_actor": {k: float(np.mean(ref[k])) for k in KEYS}}
    for mode in ("measured", "target"):
      pe, diag = run_kit(d, ckpt, task, a.seed, a.preset, cfg, mode, 0.01)
      dt = pe["turns"] - ref["turns"]
      res[f"kit_reset_{mode}"] = {**{k: float(np.mean(pe[k])) for k in KEYS}, "turns_minus_reference": float(dt.mean()),
                                  "turns_diff_ci95": float(1.96 * dt.std(ddof=1) / np.sqrt(len(dt))), **diag}
    allres[key] = res
    a.out.write_text(json.dumps(allres, indent=1) + "\n")
    km, kt = res["kit_reset_measured"], res["kit_reset_target"]
    print(f"[{key}] ref {res['reference_training_actor']['turns']:.3f} | kit measured {km['turns']:.3f} "
          f"({km['turns_minus_reference']:+.3f}±{km['turns_diff_ci95']:.3f}) clip {km['bridge_clip_fraction_of_steps']:.4f} | "
          f"kit target {kt['turns']:.3f} ({kt['turns_minus_reference']:+.3f}±{kt['turns_diff_ci95']:.3f}) "
          f"obs err {kt['obs_max_abs_err_after_10_steps']}", flush=True)


if __name__ == "__main__":
  main()
