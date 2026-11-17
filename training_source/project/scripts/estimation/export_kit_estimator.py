#!/usr/bin/env python3
"""Export the frozen cube-state estimator (B) for the torch-free LEAP deploy kit.

Writes ``cube_estimator.npz`` with everything Condition C's actor needs on top
of the 320-D observation: the estimator MLP (Linear + LayerNorm + ELU trunk,
19-output head), its normalisation buffers, the 24 cube-symmetry matrices in
the exact order used in training, and the fixed feature scales of
``scratch_study/features.py``.  A probe batch with the training-side features
is stored too; the kit re-runs it on every load and refuses to start if its
NumPy features differ.

    python scripts/estimation/export_kit_estimator.py --out ../../LEAP_Hand_RL_Deploy_Kit/policies/C_seed2/cube_estimator.npz
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from in_hand_rotation_mjlab.estimation.model import load_estimator  # noqa: E402
from in_hand_rotation_mjlab.estimation.rotations import cube_symmetry_matrices  # noqa: E402
from in_hand_rotation_mjlab.scratch_study import features as F  # noqa: E402

DEFAULT = PROJECT / "experiments" / "scratch_state_estimation" / "models" / "estimator_B.pt"


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--estimator", type=Path, default=DEFAULT)
  p.add_argument("--out", type=Path, required=True)
  a = p.parse_args()

  est = load_estimator(a.estimator).eval()
  cfg = est.cfg
  if not (cfg.layer_norm and cfg.activation == "elu" and cfg.dropout == 0 and len(cfg.hidden_dims) == 3):
    raise ValueError(f"unsupported estimator config {cfg}")
  sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in est.state_dict().items()}
  arrays = {
    "obs_mean": sd["obs_mean"], "obs_std": sd["obs_std"],
    "pos_mean": sd["pos_mean"], "pos_std": sd["pos_std"],
    "lin_vel_std": sd["lin_vel_std"], "ang_vel_std": sd["ang_vel_std"],
    "head_w": sd["head.weight"], "head_b": sd["head.bias"],
  }
  for k, (lin, ln) in enumerate(((0, 1), (3, 4), (6, 7))):
    arrays[f"w{k}"], arrays[f"b{k}"] = sd[f"trunk.{lin}.weight"], sd[f"trunk.{lin}.bias"]
    arrays[f"ln{k}_w"], arrays[f"ln{k}_b"] = sd[f"trunk.{ln}.weight"], sd[f"trunk.{ln}.bias"]
  ln_eps = {m.eps for m in est.trunk if isinstance(m, torch.nn.LayerNorm)}
  assert len(ln_eps) == 1

  g = torch.Generator().manual_seed(0)
  probe = est.obs_mean + est.obs_std * torch.randn(64, cfg.obs_dim, generator=g)
  with torch.no_grad():
    feats = F.features_from_estimator(est, probe)
  arrays.update(
    format=np.array("leap_kit_cube_estimator/v1"),
    source=np.array(a.estimator.name),
    source_sha256=np.array(hashlib.sha256(a.estimator.read_bytes()).hexdigest()),
    ln_eps=np.array(float(ln_eps.pop()), np.float32),
    sym=cube_symmetry_matrices(dtype=torch.float32).numpy(),
    feat_p_mean=np.array(F.P_MEAN, np.float32), feat_p_std=np.array(F.P_STD, np.float32),
    feat_v_std=np.array(F.V_STD, np.float32), feat_w_std=np.array(F.W_STD, np.float32),
    feature_names=np.array(F.FEATURE_NAMES),
    probe_obs=probe.numpy().astype(np.float32), probe_features=feats.numpy().astype(np.float32),
  )
  a.out.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(a.out, **arrays)
  print(f"[export] {a.estimator.name} (sha256 {arrays['source_sha256']}) -> {a.out}")


if __name__ == "__main__":
  main()
