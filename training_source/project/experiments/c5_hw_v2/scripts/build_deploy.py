#!/usr/bin/env python3
"""C5-HW-V2 deployment candidate (copy of c5_hw_v1/scripts/build_deploy.py), offline only (no hardware, no motor
commands).  Built whatever the verdict (instruction 2026-11-26, Task 2D) and LABELLED with it in MANIFEST.txt:
"ready for supervisor review" only after a single-seed pass, otherwise "not recommended".
Builds a kit-format folder deploy/policies/C5HW2_seed1/ (own checkpoint + params),
verifies the kit's NumPy runtime (policy_runtime.LeapPolicy, unchanged) against the RSL-RL training actor,
checks the action mapping / limits integration exactly as grasp_rotation_v2/scripts/build_deploy.py, and
compares the kit description with the original C5 deployment folder (grasp_rotation_v2/deploy/policies/C5_seed1,
read only).  Writes deploy/MANIFEST.txt and results/deploy_verification.json.

    python experiments/c5_hw_v2/scripts/build_deploy.py
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[3]
E = PROJECT / "experiments" / "c5_hw_v2"
C5_DEPLOY = PROJECT / "experiments" / "grasp_rotation_v2" / "deploy" / "policies" / "C5_seed1"
_spec = importlib.util.spec_from_file_location("hw_build_deploy", PROJECT / "experiments" / "hwlimit_ac" / "scripts" / "build_deploy.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)  # also puts the kit directory on sys.path


def main():
  cmp = json.loads((E / "results" / "v2_comparison.json").read_text())
  verdict = cmp.get("verdict", "")
  label = "ready for supervisor review" if verdict.startswith("SINGLE-SEED PASS") else "NOT RECOMMENDED (verdict: %s)" % verdict
  from tensordict import TensorDict

  from policy_runtime import LeapPolicy

  from in_hand_rotation_mjlab.estimation.layout import JOINT_NAMES
  from in_hand_rotation_mjlab.hwlimit_study.limits import approved_command_limits, kit_limits_sim_order

  conv = json.loads((E / "results" / "convergence.json").read_text())["runs"]["C5HW2_1"]
  it = conv["selected_iteration"]
  src = E / "checkpoints" / f"C5HW2_1_selected_model_{it}.pt"
  dst = E / "deploy" / "policies" / "C5HW2_seed1"
  if dst.exists():
    sys.exit(f"{dst} exists; not overwritten")
  (dst / "params").mkdir(parents=True)
  ck = dst / f"model_{it}.pt"
  shutil.copy2(src, ck)
  for f in ("env.yaml", "agent.yaml"):
    shutil.copy2(E / "checkpoints" / "C5HW2_1_params" / f, dst / "params" / f)
  soft_lo = np.array([-0.250, -0.995, -0.446, -0.306] * 3 + [-0.288, -0.288, -0.397, -1.260])
  soft_hi = np.array([2.166, 0.995, 1.825, 1.982] * 3 + [2.033, 2.033, 2.370, 1.799])
  train_lo, train_hi = approved_command_limits(soft_lo, soft_hi)
  kit_lo, kit_hi = kit_limits_sim_order()
  g = torch.Generator().manual_seed(7)
  pol = LeapPolicy(dst)
  d = pol.describe()
  ref, sd = B.reference_actor(ck, "A")
  obs = sd["obs_normalizer._mean"] + sd["obs_normalizer._std"] * torch.randn(512, 320, generator=g)
  with torch.no_grad():
    u_ref = ref(TensorDict({"actor": obs}, batch_size=[512])).numpy()
  on = obs.numpy().astype(np.float32)
  u_kit = np.stack([pol.actor(o, None) for o in on])
  prev = np.random.default_rng(1 + 200).uniform(train_lo - 0.05, train_hi + 0.05, size=(4096, 16)).clip(train_lo, train_hi)
  u = np.random.default_rng(1 + 9).normal(0, 2, size=(4096, 16))
  delta = (1 / 24) * np.tanh(u)
  t_train = np.clip(prev + delta, train_lo, train_hi)
  t_kit = np.clip(np.clip(prev + delta, pol.lower_limits, pol.upper_limits), kit_lo, kit_hi)
  diff = np.abs(t_kit - t_train)
  c5 = LeapPolicy(C5_DEPLOY).describe()
  desc = {k: (x.tolist() if isinstance(x, np.ndarray) else x) for k, x in d.items()}
  c5desc = {k: (x.tolist() if isinstance(x, np.ndarray) else x) for k, x in c5.items()}
  differing = sorted(k for k in set(desc) | set(c5desc) if desc.get(k) != c5desc.get(k))
  rec = {"status": label, "verdict": verdict, "checkpoint": ck.name, "checkpoint_sha256": B.sha(ck), "source_checkpoint_sha256": B.sha(src), "selected_iteration": it,
         "kit_describe": desc, "max_abs_raw_action_diff": float(np.abs(u_kit - u_ref).max()),
         "max_abs_tanh_action_diff": float(np.abs(np.tanh(u_kit) - np.tanh(u_ref)).max()),
         "estimator_present": pol.estimator is not None, "obs_dim": int(on.shape[1]),
         "default_pose_matches_training": bool(np.allclose(pol.default_joint_pos,
                                                           [0.1, -0.4, 1.3, 0.0, 0.1, 0.0, 1.3, 0.0, 0.1, 0.4, 1.3, 0.0, 1.45, 1.5, 0.579, 1.37], atol=1e-6)),
         "integration_max_diff_vs_training_rad": float(diff.max()),
         "integration_joints_differing": sorted({JOINT_NAMES[j] for j in np.argwhere(diff > 1e-6)[:, 1]}),
         "kit_describe_fields_differing_from_C5_seed1": differing}
  rec["kit_equals_training_within_3e-5"] = bool(rec["max_abs_raw_action_diff"] <= 3e-5)  # instruction 2026-11-26
  assert rec["max_abs_tanh_action_diff"] < 1e-4 and rec["kit_equals_training_within_3e-5"], rec
  assert not rec["estimator_present"] and d.get("action_squash") == "tanh", d
  lines = ["# C5-HW-V2 deployment candidate (kit format, offline-verified only; single-seed exploratory); scripts/build_deploy.py", f"# status: {label}",
           "# NOT hardware-approved: any powered test needs a separate supervisor-approved, supervised session", "# folder | file | sha256"]
  for p in sorted((E / "deploy" / "policies").rglob("*")):
    if p.is_file():
      lines.append(f"{p.parent.relative_to(E / 'deploy' / 'policies')} | {p.name} | {B.sha(p)}")
  (E / "deploy" / "MANIFEST.txt").write_text("\n".join(lines) + "\n")
  (E / "results" / "deploy_verification.json").write_text(json.dumps({"policies": {dst.name: rec}}, indent=1))
  print(json.dumps({k: v for k, v in rec.items() if k != "kit_describe"}, indent=1))


if __name__ == "__main__":
  main()
