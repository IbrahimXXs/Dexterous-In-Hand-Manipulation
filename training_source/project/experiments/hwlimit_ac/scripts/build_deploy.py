#!/usr/bin/env python3
"""Phase 6: deployment-compatible policy folders + offline verification (no hardware).

For each run's selected checkpoint, writes ``deploy/policies/<C>_seed<s>/`` in the deploy
kit's format (checkpoint, ``params/env.yaml``, ``params/agent.yaml`` and, for C,
``cube_estimator.npz`` exported from the frozen estimator B), outside the kit directory
(the kit takes any folder: ``./leap.sh shadow <folder>``).  The kit itself is not modified.

Verifies, CPU only:
  1. the kit runtime's parsed interface (``LeapPolicy.describe()``): obs terms and
     history, tanh squash, delta range +-1/24 rad, clip_actions null, default pose equal
     to training, estimator present for C only;
  2. the kit's NumPy network (``LeapPolicy.actor`` + ``CubeEstimatorNP``) against the
     RSL-RL training actor on 512 probe observations (raw action, tanh action, features);
  3. the kit's single-step integration against the training action term on probe
     commands: target = clip(prev + delta(tanh u)) with the kit's own limits followed by
     the kit's hardware clip, vs. the training clamp (approved limits) -- reports where
     the two differ (expected: only the PIP upper bound, soft limit 1.825 in training vs.
     the model's hard limit 1.885 in the kit runtime).

    python experiments/hwlimit_ac/scripts/build_deploy.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[3]
REPO = PROJECT.parents[1]
E = PROJECT / "experiments" / "hwlimit_ac"
S = PROJECT / "experiments" / "scratch_state_estimation"
KIT = REPO / "LEAP_Hand_RL_Deploy_Kit" / "kit"
EST = S / "models" / "estimator_B.pt"
sys.path.insert(0, str(KIT))


def sha(p: Path) -> str:
  return hashlib.sha256(p.read_bytes()).hexdigest()


def reference_actor(ckpt: Path, cond: str):
  from rsl_rl.models.mlp_model import MLPModel
  from tensordict import TensorDict

  from in_hand_rotation_mjlab.scratch_study.actors import EstimatedStateMLPModel

  sd = torch.load(ckpt, map_location="cpu", weights_only=False)["actor_state_dict"]
  obs = TensorDict({"actor": torch.zeros(1, 320)}, batch_size=[1])
  kw = dict(hidden_dims=[512, 512, 256], activation="elu", obs_normalization=True, stochastic=True,
            init_noise_std=0.7, noise_std_type="scalar")
  if cond == "C":
    actor = EstimatedStateMLPModel(obs, {"actor": ["actor"]}, "actor", 16, estimator_path=str(EST), **kw)
  else:
    actor = MLPModel(obs, {"actor": ["actor"]}, "actor", 16, **kw)
  actor.load_state_dict(sd, strict=True)
  return actor.eval(), sd


def main():
  from tensordict import TensorDict

  from policy_runtime import LeapPolicy

  from in_hand_rotation_mjlab.hwlimit_study.limits import approved_command_limits, kit_limits_sim_order

  conv = json.loads((E / "results" / "convergence.json").read_text())
  out_root = E / "deploy" / "policies"
  out_root.mkdir(parents=True, exist_ok=True)
  npz = E / "deploy" / "cube_estimator.npz"
  if not npz.exists():
    subprocess.run([sys.executable, str(PROJECT / "scripts" / "estimation" / "export_kit_estimator.py"), "--out", str(npz)],
                   check=True, cwd=PROJECT)
  kit_npz = REPO / "LEAP_Hand_RL_Deploy_Kit" / "policies" / "C_seed2" / "cube_estimator.npz"
  report = {"cube_estimator_npz_sha256": sha(npz), "kit_C_seed2_npz_sha256": sha(kit_npz) if kit_npz.exists() else None,
            "policies": {}}
  soft_lo = np.array([-0.250, -0.995, -0.446, -0.306] * 3 + [-0.288, -0.288, -0.397, -1.260])
  soft_hi = np.array([2.166, 0.995, 1.825, 1.982] * 3 + [2.033, 2.033, 2.370, 1.799])
  train_lo, train_hi = approved_command_limits(soft_lo, soft_hi)
  kit_lo, kit_hi = kit_limits_sim_order()
  g = torch.Generator().manual_seed(7)
  for run, v in conv["runs"].items():
    cond, seed, it = run[0], run[1], v["selected_iteration"]
    src_dir = Path(json.loads((E / "results" / "curves" / f"{run}.json").read_text())["run_dir"])
    if not src_dir.is_absolute():
      src_dir = PROJECT / src_dir
    dst = out_root / f"{cond}_seed{seed}"
    (dst / "params").mkdir(parents=True, exist_ok=True)
    ck = dst / f"model_{it}.pt"
    shutil.copy2(src_dir / f"model_{it}.pt", ck)
    for f in ("env.yaml", "agent.yaml"):
      shutil.copy2(src_dir / "params" / f, dst / "params" / f)
    if cond == "C":
      shutil.copy2(npz, dst / "cube_estimator.npz")
    pol = LeapPolicy(dst)
    d = pol.describe()
    ref, sd = reference_actor(ck, cond)
    # probe observations around the training normaliser
    obs = sd["obs_normalizer._mean"] + sd["obs_normalizer._std"] * torch.randn(512, 320, generator=g)
    with torch.no_grad():
      u_ref = ref(TensorDict({"actor": obs}, batch_size=[512])).numpy()
      f_ref = ref.last_features.numpy() if cond == "C" else None
    on = obs.numpy().astype(np.float32)
    feats = np.stack([pol.estimator.features(o) for o in on]) if pol.estimator is not None else None
    u_kit = np.stack([pol.actor(o, None if feats is None else feats[i]) for i, o in enumerate(on)])
    # single-step integration on probe commands near / beyond the limits
    prev = np.random.default_rng(int(seed)).uniform(train_lo - 0.05, train_hi + 0.05, size=(4096, 16)).clip(train_lo, train_hi)
    u = np.random.default_rng(int(seed) + 9).normal(0, 2, size=(4096, 16))
    delta = (1 / 24) * np.tanh(u)
    t_train = np.clip(prev + delta, train_lo, train_hi)
    t_kit = np.clip(np.clip(prev + delta, pol.lower_limits, pol.upper_limits), kit_lo, kit_hi)
    diff = np.abs(t_kit - t_train)
    from in_hand_rotation_mjlab.estimation.layout import JOINT_NAMES

    rec = {
      "checkpoint": ck.name, "checkpoint_sha256": sha(ck), "selected_iteration": it,
      "kit_describe": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()},
      "max_abs_raw_action_diff": float(np.abs(u_kit - u_ref).max()),
      "max_abs_tanh_action_diff": float(np.abs(np.tanh(u_kit) - np.tanh(u_ref)).max()),
      "max_abs_feature_diff": float(np.abs(feats - f_ref).max()) if feats is not None else None,
      "estimator_present": pol.estimator is not None,
      "default_pose_matches_training": bool(np.allclose(pol.default_joint_pos,
                                                        [0.1, -0.4, 1.3, 0.0, 0.1, 0.0, 1.3, 0.0, 0.1, 0.4, 1.3, 0.0, 1.45, 1.5, 0.579, 1.37], atol=1e-6)),
      "integration_max_diff_vs_training_rad": float(diff.max()),
      "integration_joints_differing": sorted({JOINT_NAMES[j] for j in np.argwhere(diff > 1e-6)[:, 1]}),
      "kit_runtime_limits": {n: [float(l), float(h)] for n, l, h in zip(JOINT_NAMES, pol.lower_limits, pol.upper_limits)},
    }
    assert rec["max_abs_tanh_action_diff"] < 1e-4, rec
    assert rec["estimator_present"] == (cond == "C")
    assert d.get("action_squash") == "tanh", d
    report["policies"][f"{cond}_seed{seed}"] = rec
    print(f"{cond}_seed{seed} model_{it}: raw diff {rec['max_abs_raw_action_diff']:.2e} tanh diff "
          f"{rec['max_abs_tanh_action_diff']:.2e} feat diff {rec['max_abs_feature_diff']} | integration differs on "
          f"{rec['integration_joints_differing']} (max {rec['integration_max_diff_vs_training_rad']:.3f} rad)", flush=True)
  lines = ["# Deployment folders (kit format) for the selected checkpoints; built by scripts/build_deploy.py", "# folder | file | sha256"]
  for p in sorted(out_root.rglob("*")):
    if p.is_file():
      lines.append(f"{p.parent.relative_to(out_root)} | {p.name} | {sha(p)}")
  (E / "deploy" / "MANIFEST.txt").write_text("\n".join(lines) + "\n")
  (E / "results" / "deploy_verification.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
  main()
