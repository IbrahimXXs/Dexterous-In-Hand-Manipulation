#!/usr/bin/env python3
"""Deployment readiness, offline (no hardware): kit-format folders for the selected
Grasp-Rotation-V2 checkpoints (P0, C5) in ``deploy/policies/<cond>_seed<s>/`` and verification
of the kit's NumPy runtime against the RSL-RL training actor.  Same procedure as
``experiments/retention_v1/scripts/build_deploy.py`` (reuses ``hwlimit_ac/scripts/build_deploy.py``).

    python experiments/grasp_rotation_v2/scripts/build_deploy.py
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[3]
E = PROJECT / "experiments" / "grasp_rotation_v2"
_spec = importlib.util.spec_from_file_location("hw_build_deploy", PROJECT / "experiments" / "hwlimit_ac" / "scripts" / "build_deploy.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)  # also puts the kit directory on sys.path


def main():
  from tensordict import TensorDict

  from policy_runtime import LeapPolicy

  from in_hand_rotation_mjlab.estimation.layout import JOINT_NAMES
  from in_hand_rotation_mjlab.hwlimit_study.limits import approved_command_limits, kit_limits_sim_order

  conv = json.loads((E / "results" / "convergence.json").read_text())["runs"]
  out_root = E / "deploy" / "policies"
  out_root.mkdir(parents=True, exist_ok=True)
  soft_lo = np.array([-0.250, -0.995, -0.446, -0.306] * 3 + [-0.288, -0.288, -0.397, -1.260])
  soft_hi = np.array([2.166, 0.995, 1.825, 1.982] * 3 + [2.033, 2.033, 2.370, 1.799])
  train_lo, train_hi = approved_command_limits(soft_lo, soft_hi)
  kit_lo, kit_hi = kit_limits_sim_order()
  g = torch.Generator().manual_seed(7)
  report = {"policies": {}}
  for run, v in conv.items():  # "P0_1", "C5_1"
    cond, seed = run.split("_")
    it = v["selected_iteration"]
    src = E / "checkpoints" / f"{run}_selected_model_{it}.pt"
    dst = out_root / f"{cond}_seed{seed}"
    (dst / "params").mkdir(parents=True, exist_ok=True)
    ck = dst / f"model_{it}.pt"
    shutil.copy2(src, ck)
    for f in ("env.yaml", "agent.yaml"):
      shutil.copy2(E / "checkpoints" / f"{run}_params" / f, dst / "params" / f)
    pol = LeapPolicy(dst)
    d = pol.describe()
    ref, sd = B.reference_actor(ck, "A")
    obs = sd["obs_normalizer._mean"] + sd["obs_normalizer._std"] * torch.randn(512, 320, generator=g)
    with torch.no_grad():
      u_ref = ref(TensorDict({"actor": obs}, batch_size=[512])).numpy()
    on = obs.numpy().astype(np.float32)
    u_kit = np.stack([pol.actor(o, None) for o in on])
    prev = np.random.default_rng(int(seed) + (0 if cond == 'P0' else 100)).uniform(train_lo - 0.05, train_hi + 0.05, size=(4096, 16)).clip(train_lo, train_hi)
    u = np.random.default_rng(int(seed) + 9).normal(0, 2, size=(4096, 16))
    delta = (1 / 24) * np.tanh(u)
    t_train = np.clip(prev + delta, train_lo, train_hi)
    t_kit = np.clip(np.clip(prev + delta, pol.lower_limits, pol.upper_limits), kit_lo, kit_hi)
    diff = np.abs(t_kit - t_train)
    rec = {
      "checkpoint": ck.name, "checkpoint_sha256": B.sha(ck), "selected_iteration": it,
      "kit_describe": {k: (x.tolist() if isinstance(x, np.ndarray) else x) for k, x in d.items()},
      "max_abs_raw_action_diff": float(np.abs(u_kit - u_ref).max()),
      "max_abs_tanh_action_diff": float(np.abs(np.tanh(u_kit) - np.tanh(u_ref)).max()),
      "estimator_present": pol.estimator is not None,
      "default_pose_matches_training": bool(np.allclose(pol.default_joint_pos,
                                                        [0.1, -0.4, 1.3, 0.0, 0.1, 0.0, 1.3, 0.0, 0.1, 0.4, 1.3, 0.0, 1.45, 1.5, 0.579, 1.37], atol=1e-6)),
      "integration_max_diff_vs_training_rad": float(diff.max()),
      "integration_joints_differing": sorted({JOINT_NAMES[j] for j in np.argwhere(diff > 1e-6)[:, 1]}),
    }
    assert rec["max_abs_tanh_action_diff"] < 1e-4, rec
    assert not rec["estimator_present"] and d.get("action_squash") == "tanh", d
    report["policies"][dst.name] = rec
    print(f"{dst.name} model_{it}: raw diff {rec['max_abs_raw_action_diff']:.2e} tanh diff {rec['max_abs_tanh_action_diff']:.2e} "
          f"default pose ok {rec['default_pose_matches_training']} | integration differs on {rec['integration_joints_differing']} "
          f"(max {rec['integration_max_diff_vs_training_rad']:.3f} rad)", flush=True)
  lines = ["# Deployment folders (kit format) for the selected Grasp-Rotation-V2 checkpoints (P0, C5); scripts/build_deploy.py", "# folder | file | sha256"]
  for p in sorted(out_root.rglob("*")):
    if p.is_file():
      lines.append(f"{p.parent.relative_to(out_root)} | {p.name} | {B.sha(p)}")
  (E / "deploy" / "MANIFEST.txt").write_text("\n".join(lines) + "\n")
  (E / "results" / "deploy_verification.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
  main()
