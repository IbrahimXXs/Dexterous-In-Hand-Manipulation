#!/usr/bin/env python3
"""Export the fingertip force estimator for the torch-free LEAP deploy kit.

Writes ``force_estimator.npz``: the MLP (Linear + LayerNorm + ELU trunk,
hurdle or direct head), its input normalisation and per-finger force scales,
plus a probe batch with the training-side outputs.  The kit re-runs the probe
on every load and refuses to work if its NumPy forces differ.

    python scripts/estimation/export_kit_force_estimator.py \
        --out ../../LEAP_Hand_RL_Deploy_Kit/monitors/force_estimator.npz
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

from in_hand_rotation_mjlab.force_estimation.model import load_force_estimator  # noqa: E402

DEFAULT = PROJECT / "experiments" / "force_estimation" / "models" / "force_estimator.pt"


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--checkpoint", type=Path, default=DEFAULT)
  p.add_argument("--out", type=Path, required=True)
  a = p.parse_args()

  m = load_force_estimator(a.checkpoint).eval()
  cfg = m.cfg
  if not (cfg.layer_norm and len(cfg.hidden_dims) == 3):
    raise ValueError(f"unsupported force-estimator config {cfg}")
  sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in m.state_dict().items()}
  arrays = {
    "obs_mean": sd["obs_mean"], "obs_std": sd["obs_std"], "force_scale": sd["force_scale"],
    "head_w": sd["head.weight"], "head_b": sd["head.bias"],
  }
  for k, (lin, ln) in enumerate(((0, 1), (3, 4), (6, 7))):
    arrays[f"w{k}"], arrays[f"b{k}"] = sd[f"trunk.{lin}.weight"], sd[f"trunk.{lin}.bias"]
    arrays[f"ln{k}_w"], arrays[f"ln{k}_b"] = sd[f"trunk.{ln}.weight"], sd[f"trunk.{ln}.bias"]
  ln_eps = {mod.eps for mod in m.trunk if isinstance(mod, torch.nn.LayerNorm)}
  assert len(ln_eps) == 1

  g = torch.Generator().manual_seed(0)
  probe = m.obs_mean + m.obs_std * torch.randn(64, cfg.obs_dim, generator=g)
  with torch.no_grad():
    forces = m(probe)
    pc = m.contact_probability(probe)
  arrays.update(
    format=np.array("leap_kit_force_estimator/v1"),
    head=np.array(cfg.head),
    finger_order=np.array(cfg.finger_order),
    source=np.array(a.checkpoint.name),
    source_sha256=np.array(hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()),
    ln_eps=np.array(float(ln_eps.pop()), np.float32),
    probe_obs=probe.numpy().astype(np.float32),
    probe_forces=forces.numpy().astype(np.float32),
    probe_contact_prob=(pc if pc is not None else torch.full_like(forces, np.nan)).numpy().astype(np.float32),
  )
  a.out.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(a.out, **arrays)
  print(f"[export] {a.checkpoint.name} ({cfg.head} head, sha256 {arrays['source_sha256']}) -> {a.out}")


if __name__ == "__main__":
  main()
